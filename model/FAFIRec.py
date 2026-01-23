# model/wmk_rec.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class FAFIRec(nn.Module):
    """
    Weighted-MMD alignment + Gated fusion + SASRec-style user encoder.
    Required interfaces:
      - forward(sample_items_id, sample_items_content, log_mask, device) -> loss
      - user_encoder(input_embs, log_mask, device) -> [B, L, D]
      - get_item_embeddings(item_word_embs, use_modal, device) -> [N, D]
    """
    def __init__(self, args, item_num, use_modal=True, item_word_embs=None):
        super().__init__()
        self.item_num = item_num + 1  # Reserve 0 as padding
        self.use_modal = use_modal
        self.d_model = args.embedding_dim
        self.dropout = getattr(args, "dropout", 0.2)
        self.n_layers = getattr(args, "n_layers", 2)
        self.n_heads = getattr(args, "n_heads", 2)
        self.neg_k = getattr(args, "neg_k", 50)
        self.align_lambda = getattr(args, "align_lambda", 0.1)  # MMD loss weight
        self.gate_lambda  = getattr(args, "gate_lambda", 1.0)  # Gate fusion weight in loss (optional)
        self.mmd_gamma = getattr(args, "mmd_gamma", 1.0)       # Reciprocal of RBF kernel bandwidth
        self.alpha_rel = getattr(args, "alpha_rel", 0.5)       # User reliability balance coefficient
        self.max_seq_len = getattr(args, "max_seq_len", 50)       # User reliability balance coefficient
        self.batch_size = getattr(args, "batch_size", 128)       # User reliability balance coefficient

        # ----------- Item tables -----------
        self.item_id_table = nn.Embedding(self.item_num, self.d_model, padding_idx=0)
        nn.init.normal_(self.item_id_table.weight, std=0.02)

        # Text features -> d_model
        self.has_content = (item_word_embs is not None)
        self.content_in_dim = None
        if item_word_embs is not None and use_modal:
            if isinstance(item_word_embs, torch.Tensor):
                self.content_in_dim = item_word_embs.shape[-1]
            else:
                raise ValueError("item_word_embs should be torch.Tensor if provided.")
        if self.content_in_dim is None:
            # If not provided during initialization, will be determined based on input content in get_item_embeddings during training/evaluation
            self.content_in_dim = self.d_model
        self.proj_content = nn.Linear(self.content_in_dim, self.d_model, bias=False)

        # Global and dimension-wise gating
        self.gate_global = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )
        self.gate_dim = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.Sigmoid()
        )

        # ----------- SASRec-style encoder -----------
        self.pos_emb = nn.Embedding(self.max_seq_len + 1, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.ln = nn.LayerNorm(self.d_model)

        # Prediction head (using dot product with item table for softmax, more memory efficient)
        # Optional to add a temperature parameter
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        # Loss choices
        self.use_bpr = getattr(args, "use_bpr", True)
        bce_pos_weight = getattr(args, "bce_pos_weight", None)  # Optional: weight for positive sample imbalance
        if bce_pos_weight is not None:
            # Support float or Tensor
            if not torch.is_tensor(bce_pos_weight):
                bce_pos_weight = torch.tensor(float(bce_pos_weight))
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight, reduction="mean")
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction="mean")

    # ----------- Weighted MMD -----------
    def _rbf_kernel(self, x, y, gamma):
        # x: [B, D], y: [B, D]
        xx = (x*x).sum(dim=1, keepdim=True)  # [B,1]
        yy = (y*y).sum(dim=1, keepdim=True)  # [B,1]
        xy = x @ y.t()                        # [B,B]
        dist2 = xx + yy.t() - 2*xy
        return torch.exp(-gamma * torch.clamp(dist2, min=0.0))

    def _weighted_mmd(self, z_id, z_txt, w):
        # z_id/z_txt: [B,D], w: [B] in [0,1]
        # 采用样本权重的MMD^2: E_w[k(x,x)] + E_w[k(y,y)] - 2 E_w[k(x,y)]
        Kxx = self._rbf_kernel(z_id, z_id, self.mmd_gamma)
        Kyy = self._rbf_kernel(z_txt, z_txt, self.mmd_gamma)
        Kxy = self._rbf_kernel(z_id, z_txt, self.mmd_gamma)
        # 归一化权重
        w = w.detach()
        w = w / (w.sum() + 1e-12)
        W = torch.outer(w, w)  # [B,B]
        mmd2 = (W * Kxx).sum() + (W * Kyy).sum() - 2.0 * (W * Kxy).sum()
        return mmd2

    # ----------- User reliability estimation -----------
    def _user_reliability(self, seq_len_norm, id_emb_mean, txt_emb_mean):
        # seq_len_norm: [B] normalized history length; cosine similarity measures modality consistency
        cos = F.cosine_similarity(id_emb_mean, txt_emb_mean, dim=-1)  # [-1,1]
        modal_agree = (cos + 1.0) / 2.0                               # [0,1]
        # Fusion: alpha * collaborative strength + (1-alpha) * modality consistency
        rel = self.alpha_rel * seq_len_norm + (1 - self.alpha_rel) * modal_agree
        return rel.clamp(0, 1)

    # ----------- Gated fusion -----------
    def _fuse_items(self, e_id, e_txt, u_rel):
        # e_id/e_txt: [N,D], u_rel: scalar or [N], used to control id/text ratio
        if isinstance(u_rel, torch.Tensor) and u_rel.dim() == 1:
            rel = u_rel.unsqueeze(-1)  # [N,1]
        else:
            rel = torch.tensor(u_rel, device=e_id.device).view(1, 1)
        pair = torch.cat([e_id, e_txt], dim=-1)                # [N,2D]
        g_global = self.gate_global(pair)                      # [N,1]
        g_dim    = self.gate_dim(pair)                         # [N,D]
        # Final gate: modulated by both global gate and user reliability
        g = (g_global * g_dim) * rel                           # [N,D]
        return g * e_txt + (1 - g) * e_id                      # [N,D]

    # ----------- Build full item embedding table -----------
    def get_item_embeddings(self, item_word_embs, use_modal, device):
        # Return full table [item_num, d_model] for retrieval/scoring, with 0th row as 0 vector for masking
        tbl_id = self.item_id_table.weight                      # [N,D]
        if not (use_modal and item_word_embs is not None):
            out = tbl_id
        else:
            if item_word_embs.device != device:
                item_word_embs = item_word_embs.to(device)
            e_txt = self.proj_content(item_word_embs)          # [N,D] (requires same N as ID table)
            # Padding for invalid 0th position
            if e_txt.size(0) != tbl_id.size(0):
                # Assume external input is vector without 0 placeholder, add a row of 0 at the beginning
                pad0 = torch.zeros(1, e_txt.size(1), device=e_txt.device)
                e_txt = torch.cat([pad0, e_txt], dim=0)
            # Use overall average reliability for global fusion (no user in evaluation stage)
            rel_global = 0.5
            out = self._fuse_items(tbl_id, e_txt, rel_global)
        # Normalization (optional)
        return F.normalize(out, dim=-1)

    # ----------- User encoder (called during evaluation) -----------
    def user_encoder(self, input_embs, log_mask, device):
        # input_embs: [B,L,D] (from BuildEvalDataset, already item vectors)
        x = input_embs
        B, L, D = x.size()
        pos = torch.arange(1, L+1, device=device).unsqueeze(0).repeat(B,1)
        x = x + self.pos_emb(pos)
        attn_mask = self._gen_attn_mask(L, device)             # causal
        key_padding = (log_mask == 0).bool()                    # [B,L], 0 position pad
        h = self.encoder(self.ln(self.dropout_layer(x)),
                         mask=attn_mask, src_key_padding_mask=key_padding)
        return h

    def _gen_attn_mask(self, L, device):
        mask = torch.full((L, L), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    # ----------- Forward (training) -----------
    def forward(self, sample_items_id, input_embs_content, log_mask, local_rank):
        """
        Args:
            sample_items_id: [B, L, 2]  where [:, :, 0] is positive sequence, [:, :, 1] is negative sequence (same user, same timeline)
            input_embs_content: [B, L, 2, C] or None; WMKRec uses it for projection/fusion (SASRec can ignore)
            log_mask: [B, L]  1=valid, 0=padding
            local_rank: Device ID or torch.device
        Returns:
            Scalar loss
        """
        # ------ Device & Type ------
        device = (local_rank if isinstance(local_rank, torch.device)
                  else torch.device(f"cuda:{local_rank}") if isinstance(local_rank, int)
        else sample_items_id.device)
        ids = sample_items_id.long().to(device)  # [B,L,2]
        msk = log_mask.to(device)  # [B,L]

        L = self.max_seq_len + 1
        D = self.d_model

        # ------ ID representation & Text representation (degrades to all 0 if no content, does not affect gating→equivalent to ID) ------
        id_vec = self.item_id_table(ids).view(-1, L, 2, D)  # [B,L,2,D]
        if self.use_modal and (input_embs_content is not None):
            txt_raw = input_embs_content.to(device).view(-1, L, 2, self.content_in_dim)  # 期望 [B,L,2,C]
            txt_vec = self.proj_content(txt_raw)  # [B,L,2,D]
        else:
            txt_vec = torch.zeros_like(id_vec)  # No text→alignment loss and fusion will not破坏ID

        B= id_vec.shape[0]
        # ------ History/Target split (predict t+1) ------
        # History takes only the first L-1 steps of the positive branch
        hist_mask = msk  # [B,L-1]
        hist_id = id_vec[:, :-1, 0, :]  # [B,L-1,D]
        hist_txt = txt_vec[:, :-1, 0, :]  # [B,L-1,D]

        # Target (t+1) positive/negative: time-aligned to the same position
        tgt_pos_id = id_vec[:, 1:, 0, :]  # [B,L-1,D]
        tgt_pos_txt = txt_vec[:, 1:, 0, :]  # [B,L-1,D]
        tgt_neg_id = id_vec[:, 1:, 1, :]  # [B,L-1,D]
        tgt_neg_txt = txt_vec[:, 1:, 1, :]  # [B,L-1,D]

        # ------ Estimate user reliability (collaborative strength + modality consistency) ------
        seq_len = hist_mask.sum(dim=1).clamp(min=1)  # [B]
        seq_len_norm = (seq_len / float(hist_mask.size(1))).detach()
        id_mean = (hist_id * hist_mask.unsqueeze(-1)).sum(1) / seq_len.unsqueeze(-1)
        txt_mean = (hist_txt * hist_mask.unsqueeze(-1)).sum(1) / seq_len.unsqueeze(-1)
        u_rel = self._user_reliability(seq_len_norm, id_mean, txt_mean)  # [B] in [0,1]

        # ------ Gated fusion (history and target) ------
        # History: same weight for each sample, fused at token level
        fused_hist = self._fuse_items(
            hist_id.reshape(-1, D),
            hist_txt.reshape(-1, D),
            u_rel.unsqueeze(1).repeat(1, hist_id.size(1)).reshape(-1)
        ).view(B, -1, D)  # [B,L-1,D]

        # Target: aligned to the same time step with positive/negative branches
        fused_tgt_pos = self._fuse_items(
            tgt_pos_id.reshape(-1, D), tgt_pos_txt.reshape(-1, D), u_rel.repeat_interleave(hist_mask.size(1))
        ).view(B, -1, D)  # [B,L-1,D]
        fused_tgt_neg = self._fuse_items(
            tgt_neg_id.reshape(-1, D), tgt_neg_txt.reshape(-1, D), u_rel.repeat_interleave(hist_mask.size(1))
        ).view(B, -1, D)  # [B,L-1,D]

        # ------ Encode history → User representation (step by step) ------
        pos_idx = torch.arange(1, fused_hist.size(1) + 1, device=device).unsqueeze(0).repeat(B, 1)
        enc_in = fused_hist + self.pos_emb(pos_idx)
        enc_out = self.encoder(
            self.ln(self.dropout_layer(enc_in)),
            mask=self._gen_attn_mask(fused_hist.size(1), device),
            src_key_padding_mask=(hist_mask == 0).bool()
        )  # [B,L-1,D]

        # Take the representation of the "last valid position" in each sample's history
        idx_last = (seq_len - 1).long().clamp(min=0)  # [B]
        q = enc_out[torch.arange(B, device=device), idx_last, :]  # [B,D]
        q = F.normalize(q, dim=-1)

        # ------ Scoring ------
        # pos_score = (q * F.normalize(fused_tgt_pos[:, -1, :], dim=-1)).sum(-1)  # Can also use only the last target position
        # neg_score = (q * F.normalize(fused_tgt_neg[:, -1, :], dim=-1)).sum(-1)

        # Or: mean supervision for all target positions (change the above line to per-position, then aggregate by valid position mask)
        # Here following your original approach: per-position supervision (recommended). As follows:
        pos_all = (enc_out * F.normalize(fused_tgt_pos, dim=-1)).sum(-1)  # [B,L-1]
        neg_all = (enc_out * F.normalize(fused_tgt_neg, dim=-1)).sum(-1)  # [B,L-1]
        target_mask = msk.bool()  # [B,L-1]

        # ------ Loss: BPR or BCE (switch as needed) ------
        if self.use_bpr:
            pair = (pos_all - neg_all)[target_mask]
            base_loss = (-torch.log(torch.sigmoid(pair) + 1e-12)).mean() if pair.numel() > 0 else q.sum() * 0.0
        else:
            # Align with your existing BCE approach
            pos_loss = self.criterion(pos_all[target_mask], torch.ones_like(pos_all[target_mask]))
            neg_loss = self.criterion(neg_all[target_mask], torch.zeros_like(neg_all[target_mask]))
            base_loss = pos_loss + neg_loss

        # ------ WMK-MMD (weighted alignment on history tokens) ------
        z_id = hist_id[hist_mask.bool()]  # [M,D]
        z_txt = hist_txt[hist_mask.bool()]  # [M,D]
        if z_id.size(0) > 1:
            # Assign weights to valid tokens for each sample: repeat_interleave repeats each sample seq_len[b] times
            token_w = torch.repeat_interleave(u_rel, seq_len.long())
            loss_mmd = self._weighted_mmd(z_id, z_txt, token_w)
        else:
            loss_mmd = torch.tensor(0.0, device=device)

        loss = base_loss + self.align_lambda * loss_mmd
        return loss

    def get_item_tables_for_eval(self, item_word_embs, use_modal, device):
        """
        Return two tables for personalized fusion:
          A = e_id                                   [N, D]
          B = g_base ∘ (e_txt - e_id)                [N, D]
        No normalization when calling, for convenient linear combination; normalize q during scoring.
        """
        A = self.item_id_table.weight.to(device)  # [N, D]
        if not (use_modal and item_word_embs is not None):
            # No text, let B be 0 (degrade to pure ID)
            B = torch.zeros_like(A)
            return A, B

        e_txt = item_word_embs.to(device)
        if e_txt.size(0) != A.size(0):
            pad0 = torch.zeros(1, e_txt.size(1), device=device)
            e_txt = torch.cat([pad0, e_txt], dim=0)  # Align 0 position
        e_txt = self.proj_content(e_txt)  # [N, D]

        pair = torch.cat([A, e_txt], dim=-1)  # [N, 2D]
        g_global = self.gate_global(pair)  # [N, 1]
        g_dim = self.gate_dim(pair)  # [N, D]
        g_base = g_global * g_dim  # [N, D]  Independent of user

        B = g_base * (e_txt - A)  # [N, D]
        return A, B

    @torch.no_grad()
    def _ensure_eval_tables(self, item_word_embs, use_modal, device):
        """Lazy loading/caching A,B tables (A=e_id, B=g_base ∘ (e_txt - e_id))."""
        need_build = (
                not hasattr(self, "_eval_cache")
                or self._eval_cache.get("device") != device
                or self._eval_cache.get("use_modal") != bool(use_modal)
                or self._eval_cache.get("num_items") != self.item_id_table.weight.size(0)
        )
        if need_build:
            A, B = self.get_item_tables_for_eval(item_word_embs, use_modal, device)
            self._eval_cache = {
                "A": A, "B": B,
                "device": device,
                "use_modal": bool(use_modal),
                "num_items": A.size(0)
            }

    @torch.no_grad()
    def predict(self, hist_ids, device, item_word_embs=None, is_use_modal=True, topk=None, personalized=True, alpha_rel=0.5):
        """
        Batch TopK prediction (full library). Non-autoregressive.
        - hist_ids: [B,L] right-aligned, 0 padding (from Dataset)
        - filter_seen: None or List[1D LongTensor] (each user's interaction set)
        """
        self._ensure_eval_tables(item_word_embs, is_use_modal, device)
        A, B = self._eval_cache["A"], self._eval_cache["B"]

        # User vector q
        hist_ids = hist_ids.to(device)
        mask = (hist_ids != 0).long()
        hist_vec = self.item_id_table(hist_ids)  # [B, L, D]
        enc = self.user_encoder(hist_vec, mask, device)  # [B, L, D]
        last_idx = (mask.sum(dim=1).clamp(min=1) - 1).long()
        q = enc[torch.arange(hist_ids.size(0), device=device), last_idx, :]
        q = F.normalize(q, dim=-1)

        # User reliability r (personalized or constant 0.5)
        if personalized and self.use_modal and (item_word_embs is not None):
            seq_len = mask.sum(dim=1).clamp(min=1).float()
            len_strength = (seq_len / hist_ids.size(1)).clamp(0, 1)
            txt_tbl = item_word_embs.to(device)
            if txt_tbl.size(0) != self.item_id_table.weight.size(0):
                pad0 = torch.zeros(1, txt_tbl.size(1), device=device)
                txt_tbl = torch.cat([pad0, txt_tbl], dim=0)
            txt_tok = self.proj_content(txt_tbl)[hist_ids]  # [B, L, D]
            id_tok = hist_vec
            id_mean = (id_tok * mask.unsqueeze(-1)).sum(1) / seq_len.unsqueeze(-1)
            txt_mean = (txt_tok * mask.unsqueeze(-1)).sum(1) / seq_len.unsqueeze(-1)
            cos = F.cosine_similarity(id_mean, txt_mean, dim=-1)
            modal_agree = (cos + 1.0) * 0.5
            r = (alpha_rel * len_strength + (1 - alpha_rel) * modal_agree).clamp(0, 1)  # [B]
        else:
            r = torch.full((hist_ids.size(0),), 0.5, device=device)

        # Linear recombination: score = q·A^T + r * (q·B^T)
        sA = q @ A.t()  # [B, N]
        sB = q @ B.t()  # [B, N]
        scores = sA + r.unsqueeze(1) * sB  # [B, N]
        # Mask padding

        if topk is None:
            return scores
        k = int(topk)
        return torch.topk(scores, k=k, dim=1)  # (scores_topk, items_topk)
