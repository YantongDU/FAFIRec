import copy
from collections import OrderedDict
import torch.optim as optim
import re
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from data_utils import get_id_embeddings
from data_utils.utils import *
import openpyxl
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2
from torch.utils.data import Dataset
import math
import random
from tqdm import tqdm
from model.encoders import FC_Layers, User_Encoder, ADD, CAT, MLP_Layers
from model.sasrec import SASRec
from torch.nn.init import xavier_normal_
from info_nce import InfoNCE
from tllib.modules.kernels import GaussianKernel
from tllib.alignment.dan import MultipleKernelMaximumMeanDiscrepancy


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_dict(tmp_dict, num=5):
    for k, v in list(tmp_dict.items())[:num]:
        print(f"{k}: {v}")
    print('...')

def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"

def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)

def record_model_result(dataset, model_name, test_dict, note=None):
    print('Start to recorder the results.')
    rstDict = OrderedDict()
    rstDict['model'] = '\t'
    rstDict['dataset'] = '\t'
    rstDict['states'] = '\t'
    rstDict['ndcg@1'] = '\t'
    rstDict['ndcg@5'] = '\t'
    rstDict['ndcg@10'] = '\t'
    rstDict['hit@1'] = '\t'
    rstDict['hit@5'] = '\t'
    rstDict['hit@10'] = '\t'
    rstDict['note'] = '\t' if note is None else note
    rstDict['model'] = model_name
    rstDict['dataset'] = dataset

    wb = openpyxl.load_workbook("cold-start.xlsx")
    sheet = wb['Sheet1']

    for t_user_label, mean_eval in test_dict.items():
        tmp = copy.copy(rstDict)
        tmp['states'] = t_user_label
        tmp['hit@1']  = mean_eval[0]
        tmp['hit@5']  = mean_eval[1]
        tmp['hit@10'] = mean_eval[2]
        tmp['ndcg@1'] = mean_eval[3]
        tmp['ndcg@5'] = mean_eval[4]
        tmp['ndcg@10']= mean_eval[5]
        sheet.append(list(tmp.values()))

    wb.save("cold-start.xlsx")
    print('Write Done.')

@torch.no_grad()
def run_eval(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
             model, item_word_embs, users_history_for_valid, users_valid, batch_size, item_num,
             is_use_modal, mode, is_early_stop, local_rank, args):

    device = (local_rank if isinstance(local_rank, torch.device)
              else torch.device(f"cuda:{local_rank}") if isinstance(local_rank, int)
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    was_training = model.training
    model.eval()

    # Dataset & DataLoader（一步版）
    ds = EvalOneStepDataset(
        users_history_for_valid=users_history_for_valid,
        users_valid=users_valid,
        max_seq_len=getattr(args, "max_seq_len", 50),
        filter_seen=getattr(args, "filter_seen", True)
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=getattr(args, "num_workers", 2), pin_memory=True)

    topks = getattr(args, "topk", [10, 20])
    if isinstance(topks, int): topks = [topks]
    topks = sorted(set(int(k) for k in topks))
    maxK = max(topks)
    hits = {k: 0 for k in topks}
    ndcgs = {k: 0.0 for k in topks}
    total = 0

    for hist_ids, gt_items in dl:
        _, topk_items = model.predict(
            hist_ids=hist_ids, device=device,
            item_word_embs=item_word_embs, is_use_modal=is_use_modal,
            topk=maxK, personalized=True,
            alpha_rel=getattr(args, "alpha_rel", 0.5)
        )  # [B, maxK]

        gt_items = gt_items.to(device)
        total += gt_items.size(0)

        for k in topks:
            pred = topk_items[:, :k]
            hit_mask = (pred == gt_items.view(-1, 1)).any(dim=1)
            hits[k] += hit_mask.sum().item()
            if hit_mask.any():
                where = (pred == gt_items.view(-1, 1)).nonzero(as_tuple=False)
                for _, c in where:
                    ndcgs[k] += 1.0 / log2(int(c.item()) + 2.0)

    denom = float(max(1, total))
    hr_at   = {k: hits[k] / denom for k in topks}
    ndcg_at = {k: ndcgs[k] / denom for k in topks}

    main = getattr(args, "main_metric", f"ndcg@{maxK}").lower()
    if main.startswith("hr@"):
        k = int(main.split("@")[1]); eval_value = hr_at.get(k, hr_at[maxK])
    else:
        k = int(main.split("@")[1]) if "@" in main else maxK
        eval_value = ndcg_at.get(k, ndcg_at[maxK])

    improved = eval_value > max_eval_value
    need_save = improved
    if improved:
        max_eval_value = eval_value
        max_epoch = now_epoch
        early_stop_epoch = now_epoch
        early_stop_count = 0
    else:
        early_stop_count += 1
    need_break = is_early_stop and (early_stop_count >= getattr(args, "early_stop_patience", 5))

    if getattr(args, "verbose_eval", True):
        msg = [f"[Eval@epoch {now_epoch}] one-step, n={total}"]
        msg += [f"HR@{k}={hr_at[k]:.4f}" for k in topks]
        msg += [f"NDCG@{k}={ndcg_at[k]:.4f}" for k in topks]
        msg += [f"{main}:{eval_value:.4f} (best={max_eval_value:.4f} @ {max_epoch})"]
        print("  ".join(msg))

    if was_training:
        model.train()

    return max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break, need_save


def eval_model_amazon(topk, model, user_history, eval_seq, item_embeddings, test_batch_size, args, item_num, v_or_t,
                      local_rank, now_epoch, is_test=False):
    topK_list = [1, 5, 10]
    eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size, shuffle=True, num_workers=args.num_workers,
                         worker_init_fn=worker_init_fn)
    if is_test:
        desc = f"Test {now_epoch:>7}"
    else:
        desc = f"Evaluation {now_epoch:>1}"
    model.eval()
    iter_data = (
        tqdm(
            eval_dl,
            total=len(eval_dl),
            ncols=100,
            desc=set_color(desc, 'pink'),
        )
    )

    topK = topk
    item_embeddings = item_embeddings.to(local_rank)
    # print(item_embeddings.shape)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for batch_idx, data in enumerate(iter_data):
            user_ids, input_embs, log_mask, labels = data
            user_ids, input_embs, log_mask, labels = \
                user_ids.to(local_rank), input_embs.to(local_rank), \
                    log_mask.to(local_rank), labels.to(local_rank).detach()
            prec_emb = model.user_encoder(input_embs, log_mask, local_rank)[:, -1].detach()
            scores = torch.matmul(prec_emb, item_embeddings.t()).squeeze(dim=-1).detach()
            # print(scores.shape)
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topKs(score, label, item_rank, topK_list, local_rank))
        eval_all_user = torch.stack(eval_all_user, dim=0)
        mean_hitK = eval_all_user[:, 0, :].mean(dim=0).tolist()  # shape: [num_users, len(K)]
        mean_ndcgK = eval_all_user[:, 1, :].mean(dim=0).tolist()
        mean_eval = mean_hitK + mean_ndcgK
        print(v_or_t + "_methods   {}".format(
            '\t'.join([f'Hit@{k}' for k in topK_list] + [f'nDCG@{k}' for k in topK_list])))
        print_metrics_multi(mean_eval, v_or_t, topK_list)
    return mean_eval[2]


class BuildEvalDataset(Dataset):
    def __init__(self, u2seq, item_content, max_seq_len, item_num):
        self.u2seq = u2seq
        self.user_id_list = list(u2seq.keys())
        self.item_content = item_content

        self.max_seq_len = max_seq_len + 1
        self.item_num = item_num

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        user_id = self.user_id_list[user_id]
        seq = self.u2seq[user_id]
        tokens = seq[:-1]
        target = seq[-1]
        mask_len = self.max_seq_len - len(seq)
        pad_tokens = [0] * mask_len + tokens
        log_mask = [0] * mask_len + [1] * len(tokens)
        input_embs = self.item_content[pad_tokens]
        labels = np.zeros(self.item_num)
        labels[target - 1] = 1.0
        return torch.LongTensor([user_id]), \
            input_embs, \
            torch.FloatTensor(log_mask), \
            labels


def metrics_topKs(y_score, y_true, item_rank, topK_list, local_rank):
    order = torch.argsort(y_score, descending=True)
    y_true = torch.take(y_true, order)
    rank = torch.sum(y_true * item_rank).item()

    result = torch.zeros((2, len(topK_list)), device=local_rank)

    for idx, K in enumerate(topK_list):
        if rank <= K - 1:
            result[0, idx] = 1
            result[1, idx] = 1 / math.log2(rank + 1)

    return result


def print_metrics_multi(x, v_or_t, topK_list):
    K = len(topK_list)
    hit_part = '\t'.join([
        "{:0.5f}".format(x[i])
        for i, k in enumerate(topK_list)
    ])
    ndcg_part = '\t'.join([
        "{:0.5f}".format(x[i + K])
        for i, k in enumerate(topK_list)
    ])
    print(f"{v_or_t}_results\t{hit_part}\t{ndcg_part}")


def save_model(now_epoch, model, model_dir, optimizer, rng_state, cuda_rng_state, scaler):
    ckpt_path = os.path.join(model_dir, f'epoch-{now_epoch}.pt')
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': rng_state,
                'cuda_rng_state': cuda_rng_state,
                'scaler_state': scaler.state_dict()}, ckpt_path)
    print(f"Model saved to {ckpt_path}")

class BuildTrainDataset(Dataset):
    def __init__(self, u2seq, item_content,
                 item_num, max_seq_len, use_modal):
        self.u2seq = u2seq
        self.item_content = item_content
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.use_modal = use_modal

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        lable = 0
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = []
        padding_seq = [0] * mask_len_head + seq
        sample_items.append(padding_seq)
        neg_items = []
        for i in range(tokens_Len):
            sam_neg = random.randint(1, self.item_num)
            while sam_neg in seq:
                sam_neg = random.randint(1, self.item_num)
            neg_items.append(sam_neg)
        neg_items = [0] * mask_len_head + neg_items + [0]
        sample_items.append(neg_items)
        sample_items_id = torch.LongTensor(np.array(sample_items)).transpose(0, 1)

        sample_items_content = self.item_content[sample_items_id]
        return (torch.LongTensor(sample_items_id), sample_items_content, torch.FloatTensor(log_mask)), lable



class Model2_align(torch.nn.Module):

    def __init__(self, args, item_num, use_modal):
        super(Model2_align, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len + 1
        self.dnn_layers = args.dnn_layers
        self.mo_dnn_layers = args.mo_dnn_layers
        self.gamma = args.gamma
        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)
        self.turn_dim1 = FC_Layers(word_embedding_dim=args.word_embedding_dim,
                                   item_embedding_dim=args.embedding_dim,
                                   dnn_layers=2, drop_rate=args.drop_rate)


        self.fc = CAT(input_dim=args.embedding_dim * 2,
                      output_dim=args.embedding_dim,
                      drop_rate=args.drop_rate)

        self.mlp_layers = MLP_Layers(layers=[args.embedding_dim] * (self.dnn_layers + 1),
                                     dnn_layers=self.dnn_layers,
                                     drop_rate=args.drop_rate)

        self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        xavier_normal_(self.id_embedding.weight.data)
        self.criterion = nn.BCEWithLogitsLoss()
        self.loss = InfoNCE()

    def forward(self, sample_items_id, input_embs_content, log_mask, local_rank=0):
        input_embs_id = self.id_embedding(sample_items_id)
        input_embs_text = self.turn_dim1(input_embs_content)
        input_embs_all = self.fc(input_embs_id, input_embs_text)

        input_embs = input_embs_all.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        pos_items_embs = input_embs[:, :, 0]
        neg_items_embs = input_embs[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        prec_vec = self.user_encoder(input_logs_embs, log_mask, local_rank)
        pos_score = (prec_vec * target_pos_embs).sum(-1)
        neg_score = (prec_vec * target_neg_embs).sum(-1)
        pos_labels, neg_labels = torch.ones(pos_score.shape).to(local_rank), torch.zeros(neg_score.shape).to(local_rank)

        indices = torch.where(log_mask != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])

        return loss

class Model2_id(torch.nn.Module):
    def __init__(self, args, item_num, use_modal):
        super(Model2_id, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len + 1
        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)


        self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        xavier_normal_(self.id_embedding.weight.data)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, sample_items_id, input_embs_content, log_mask, local_rank):

        input_embs_all = self.id_embedding(sample_items_id)

        input_embs = input_embs_all.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        pos_items_embs = input_embs[:, :, 0]
        neg_items_embs = input_embs[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]
        prec_vec = self.user_encoder(input_logs_embs, log_mask, local_rank)
        pos_score = (prec_vec * target_pos_embs).sum(-1)
        neg_score = (prec_vec * target_neg_embs).sum(-1)
        pos_labels, neg_labels = torch.ones(pos_score.shape).to(local_rank), torch.zeros(neg_score.shape).to(local_rank)

        indices = torch.where(log_mask != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])
        return loss


class Model2_llm(torch.nn.Module):
    def __init__(self, args, item_num, use_modal, input_embs_content):
        super(Model2_llm, self).__init__()
        self.args = args
        self.args.embedding_dim = 4096
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len + 1
        self.dnn_layers = args.dnn_layers
        self.mo_dnn_layers = args.mo_dnn_layers
        self.gamma = args.gamma
        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)
        # print(args.word_embedding_dim)
        self.turn_dim1 = FC_Layers(word_embedding_dim=args.word_embedding_dim,
                                   item_embedding_dim=args.embedding_dim,
                                   dnn_layers=self.mo_dnn_layers, drop_rate=args.drop_rate)

        self.mlp_layers = MLP_Layers(layers=[args.embedding_dim] * (self.dnn_layers + 1),
                                     dnn_layers=self.dnn_layers,
                                     drop_rate=args.drop_rate)

        # self.id_embedding =  nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        self.id_embedding = nn.Embedding.from_pretrained(input_embs_content, freeze=True, padding_idx=0)
        xavier_normal_(self.id_embedding.weight.data)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, sample_items_id, input_embs_content, log_mask, local_rank):
        input_embs_all = self.id_embedding(sample_items_id)

        input_embs = input_embs_all.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        pos_items_embs = input_embs[:, :, 0]
        neg_items_embs = input_embs[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        prec_vec = self.user_encoder(input_logs_embs, log_mask, local_rank)
        pos_score = (prec_vec * target_pos_embs).sum(-1)
        neg_score = (prec_vec * target_neg_embs).sum(-1)
        pos_labels, neg_labels = torch.ones(pos_score.shape).to(local_rank), torch.zeros(neg_score.shape).to(local_rank)

        indices = torch.where(log_mask != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])
        return loss



class Model2_transfer(torch.nn.Module):

    def __init__(self, args, item_num, use_modal):
        super(Model2_transfer, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len + 1
        self.dnn_layers = args.dnn_layers
        self.mo_dnn_layers = args.mo_dnn_layers
        self.gamma = args.gamma
        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)
        self.user_enc_2 = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)
        self.turn_dim1 = FC_Layers(word_embedding_dim=args.word_embedding_dim,
                                  item_embedding_dim=args.embedding_dim,
                                  dnn_layers=self.mo_dnn_layers, drop_rate=args.drop_rate)
        self.turn_dim_random = FC_Layers(word_embedding_dim=args.word_embedding_dim,
                                  item_embedding_dim=args.embedding_dim,
                                  dnn_layers=self.mo_dnn_layers, drop_rate=args.drop_rate)
        if 'add' in args.item_tower:
            self.fc = ADD()
        elif 'cat' in args.item_tower:
            self.fc = CAT(input_dim=args.embedding_dim*2,
                          output_dim=args.embedding_dim,
                          drop_rate=args.drop_rate)
        else:
            self.fc = None

        self.mlp_layers = MLP_Layers(layers=[args.embedding_dim] * (self.dnn_layers + 1),
                              dnn_layers=self.dnn_layers,
                              drop_rate=args.drop_rate)

        self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        self.other_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        xavier_normal_(self.id_embedding.weight.data)
        self.criterion = nn.BCEWithLogitsLoss()
        #l=[-8,-4,-2,-1,0]
        self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        #kernels=[GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.)])
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])
        #l=[-8,-4,-2,-1,0]
        #kernels=[GaussianKernel(alpha=2 ** k) for k in l])
        #kernels=[LapKernel(alpha=2 ** k) for k in range(-3, 2)])
    def freeze(self):
        self.other_embedding.weight.data.copy_(self.id_embedding.weight.data)
        self.user_enc_2=copy.deepcopy(self.user_encoder)
    def forward(self, sample_items_id, input_embs_content, log_mask, local_rank):

        input_embs_id = self.other_embedding(sample_items_id)
        if self.use_modal:
            input_embs_content=self.turn_dim1(input_embs_content)
            input_embs_all = self.mlp_layers(self.fc(input_embs_id, input_embs_content))
            mkmmd_loss = 0.2*self.mkmmd_loss(input_embs_content, input_embs_id)
            #mkmmd_loss = self.gamma*(self.mkmmd_loss(input_embs_id, input_embs_content)+self.mkmmd_loss(input_embs_content, input_embs_id))
            #print('mmd loss')
            #print(mkmmd_loss)
            #input_embs_all = input_embs_id
        else:
            input_embs_all = input_embs_id
        input_embs = input_embs_all.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        pos_items_embs = input_embs[:, :, 0]
        neg_items_embs = input_embs[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        prec_vec = self.user_enc_2(input_logs_embs, log_mask, local_rank)
        pos_score = (prec_vec * target_pos_embs).sum(-1)
        neg_score = (prec_vec * target_neg_embs).sum(-1)
        pos_labels, neg_labels = torch.ones(pos_score.shape).to(local_rank), torch.zeros(neg_score.shape).to(local_rank)

        indices = torch.where(log_mask != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])
        #if self.use_modal:
        loss+=mkmmd_loss
        return loss


# eval_dataset.py
import torch
from torch.utils.data import Dataset

class EvalOneStepDataset(Dataset):
    def __init__(self, users_history_for_valid, users_valid, max_seq_len, filter_seen=True):
        self.samples = []
        self.max_seq_len = max_seq_len
        self.filter_seen = filter_seen
        for uid, hist in users_history_for_valid.items():
            hist = hist if torch.is_tensor(hist) else torch.LongTensor(hist)
            vseq = users_valid[uid]
            if len(vseq) == 0:
                continue
            gt = int(vseq[0])
            if len(hist) >= max_seq_len:
                pad_hist = hist[-max_seq_len:].clone()
            else:
                pad = torch.zeros(max_seq_len - len(hist), dtype=torch.long)
                pad_hist = torch.cat([pad, hist], dim=0)
            self.samples.append((pad_hist, gt))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--item_tower", type=str, default='modal_cat')
    parser.add_argument("--root_data_dir", type=str, default="/", )
    parser.add_argument("--dataset", type=str, default='amazon')
    parser.add_argument("--behaviors", type=str, default='amazon')
    parser.add_argument("--cold_file", type=str, default='None')
    parser.add_argument("--new_file", type=str, default='None')
    parser.add_argument("--news", type=str, default='amazon')

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5)
    parser.add_argument("--l2_weight", type=float, default=1e-5)
    parser.add_argument("--drop_rate", type=float, default=0.5)
    parser.add_argument("--dnn_layers", type=int, default=0)
    parser.add_argument("--mo_dnn_layers", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--gamma2", type=float, default=0.2)
    parser.add_argument("--gamma3", type=float, default=0.2)
    parser.add_argument("--num_k", type=int, default=0)
    parser.add_argument("--freeze", type=int, default=1)
    parser.add_argument("--left", type=int, default=-3)
    parser.add_argument("--right", type=int, default=2)
    parser.add_argument("--n_h", type=int, default=2)
    parser.add_argument("--n_v", type=int, default=2)
    parser.add_argument("--bert_model_load", type=str, default='llm')
    parser.add_argument("--freeze_paras_before", type=int, default=0)
    parser.add_argument("--word_embedding_dim", type=int, default=4096)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_attention_heads", type=int, default=2)
    parser.add_argument("--transformer_block", type=int, default=5)
    parser.add_argument("--min_seq_len", type=int, default=5)

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--load_ckpt_name", type=str, default='None')
    parser.add_argument("--label_screen", type=str, default='None')
    parser.add_argument("--logging_num", type=int, default=4)
    parser.add_argument("--testing_num", type=int, default=1)
    parser.add_argument("--local_rank", default=-1, type=int)

    parser.add_argument("--num_words_title", type=int, default=30)
    parser.add_argument("--num_words_abstract", type=int, default=50)
    parser.add_argument("--num_words_body", type=int, default=50)
    parser.add_argument("--news_attributes", type=str, default='title')

    args = parser.parse_args(args=[])
    return args
