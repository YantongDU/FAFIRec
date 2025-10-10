from utils import *
from model.F2ARec import F2ARec


# Model note information, used to identify current experiment
model_note = 'F2ARec'
# Set random seed to ensure reproducible experiments
setup_seed(2025)
# Parse command line arguments
args = parse_args()
# Set local GPU device ID
local_rank = 0

# Get available GPU count
gpus = torch.cuda.device_count()
# Whether to use modal information (text features)
is_use_modal = True  # SASRec does not use modal information
# Small value, used as threshold in certain calculations
flag=0.0001
# Set learning rate
args.lr=2.92e-04
# Build simplified directory label for saving models and logs
dir_label = f'{args.behaviors}_{model_note}'
# Build simplified log parameter string
log_paras = f'{model_note}_ed{args.embedding_dim}_lr{args.lr}_bs{args.batch_size}'

# Create model save directory
model_dir = os.path.join('./checkpoint', dir_label, log_paras)
time_run = time.strftime('-%Y%m%d-%H%M%S', time.localtime())
# Set screen display label
args.label_screen = args.label_screen + time_run

# If model directory does not exist, create it
if not os.path.exists(model_dir):
    Path(model_dir).mkdir(parents=True, exist_ok=True)

before_item_id_to_dic = {}
before_item_id_to_name = {}
before_item_name_to_id = {}
c = pd.read_csv('dataset/metadata_Prime_Pantry.csv')
c['index'] = c.index + 1
before_item_name_to_id = c.set_index('asin')['index'].to_dict()
item_num = len(before_item_name_to_id)

his = pd.read_csv('dataset/Prime_Pantry_cleaned_history.csv', converters={'history': eval})

max_seq_len = 0
min_seq_len = 100
user_seq_dic = {}
for i in range(his.shape[0]):
    user_name = his.iloc[i, 1]
    history_item_name = his.iloc[i, 2]
    item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
    user_seq_dic[user_name] = item_ids_sub_seq
    if len(item_ids_sub_seq) > max_seq_len:
        max_seq_len = len(item_ids_sub_seq)
    if len(item_ids_sub_seq) < min_seq_len:
        min_seq_len = len(item_ids_sub_seq)
args.max_seq_len = max_seq_len
args.min_seq_len = min_seq_len

user_id = 0
users_train = {}
users_valid = {}
users_test = {}
users_history_for_valid = {}
users_history_for_test = {}
user_name_to_id = {}
user_id_to_name = {}
for user_name, item_seqs in user_seq_dic.items():
    user_seq = [i for i in item_seqs]
    train = user_seq[:-2]
    valid = user_seq[-(max_seq_len + 2):-1]
    test = user_seq[-(max_seq_len + 1):]
    users_train[user_id] = train
    users_history_for_valid[user_id] = torch.LongTensor(np.array(train))
    users_valid[user_id] = valid
    users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
    users_test[user_id] = test

    user_name_to_id[user_name] = user_id
    user_id_to_name[user_id] = user_name

    user_id += 1

user_len_dict = {}
for user_id, user_seq in users_history_for_test.items():
    user_len_dict[user_id] = len(user_seq)


sorted_users = sorted(user_len_dict.items(), key=lambda x: x[1])

num_users = len(sorted_users)
front_20_percent = int(num_users * 0.2)
back_20_percent = int(num_users * 0.2)

front_users = [user_id for user_id, _ in sorted_users[:front_20_percent]]
back_users = [user_id for user_id, _ in sorted_users[-back_20_percent:]]
middle_users = [user_id for user_id, _ in sorted_users[front_20_percent:-back_20_percent]]

new_test_user_labels = ['front_20%', 'middle_60%', 'back_20%']
user_bins = {
    'front_20%': front_users,
    'middle_60%': middle_users,
    'back_20%': back_users
}

item_word_embs_tmp = torch.load('dataset/Amazon_Prime_Pantry_llm2vec.pt')
item_word_embs = torch.tensor(item_word_embs_tmp, dtype=torch.float32)
del item_word_embs_tmp

train_dataset = BuildTrainDataset(u2seq=users_train, item_content=item_word_embs, item_num=item_num,
                                  max_seq_len=max_seq_len, use_modal=is_use_modal)

train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

model_name = model_note
model = F2ARec(args, item_num, is_use_modal, item_word_embs).to(local_rank)

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)
scaler = torch.amp.GradScaler('cuda')

total_num = sum(p.numel() for p in model.parameters())
trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("##### total_num {} #####".format(total_num))
print("##### trainable_num {} #####".format(trainable_num))
len_users_train=len(users_train)
step_num = math.ceil(len_users_train / args.batch_size)
print("##### all {} steps #####".format(step_num))
steps_for_log = int(step_num / args.logging_num)
steps_for_test = int(step_num / args.testing_num)
print("##### {} logs/epoch; {} steps/log #####".format(args.logging_num, steps_for_log))
print("##### {} tests/epoch; {} steps/test #####".format(args.testing_num, steps_for_test))

setup_seed(2025)
# Initialize checkpoint related variables
checkpoint = None  # new
ckpt_path = None  # new
start_epoch = 0
# Set early stopping mechanism parameters
is_early_stop = True
max_epoch, early_stop_epoch = 0, args.epoch
max_eval_value, early_stop_count = 0, 0

for ep in range(args.epoch):
    now_epoch = ep + 1
    loss, need_break = 0.0, False
    model.train()

    iter_data = (
        tqdm(
            train_dl,
            total=len(train_dl),
            ncols=100,
            desc=set_color(f"Train {now_epoch:>6}", 'pink'),
        )
    )
    for batch_idx, data in enumerate(iter_data):
        (sample_items_id, sample_items_content, log_mask), _ = data
        sample_items_id, sample_items_content, log_mask = \
            sample_items_id.to(local_rank), sample_items_content.to(local_rank), log_mask.to(local_rank)

        sample_items_content = sample_items_content.view(-1, sample_items_content.size(-1))
        sample_items_id = sample_items_id.view(-1)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            bz_loss = model(sample_items_id, sample_items_content, log_mask, local_rank)
            loss += bz_loss.data.float()
        scaler.scale(bz_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        iter_data.set_postfix_str(set_color('batch loss: {:.5f}'.format(loss.data / (batch_idx + 1)), 'yellow'))

        # Check if loss is NaN, if so interrupt training
        if torch.isnan(loss.data):
            need_break = True
            break

    if not need_break and now_epoch % 1 == 0:
        max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break, need_save = \
            run_eval(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
                     model, item_word_embs, users_history_for_valid, users_valid, args.batch_size, item_num,
                     is_use_modal, args.mode, is_early_stop, local_rank, args)
        model.train()
        # Save model if needed
        if need_save:
            save_model(now_epoch, model, model_dir, optimizer,
                       torch.get_rng_state(), torch.cuda.get_rng_state(), scaler)
    # Break loop if training needs to be interrupted
    if need_break:
        break
save_model(now_epoch, model, model_dir, optimizer,
           torch.get_rng_state(), torch.cuda.get_rng_state(), scaler)
print('\n')
print('%' * 90)
print(' max eval Hit10 {:0.5f}  in epoch {}'.format(max_eval_value, max_epoch))
print(' early stop in epoch {}'.format(early_stop_epoch))
print('the End')
print('{} train end in epoch {}'.format(args.label_screen, early_stop_epoch))

# model = load_model(model_dir, 'epoch-{}.pt'.format(max_epoch))
load_ckpt_name = 'epoch-{}.pt'.format(max_epoch)
print('load ckpt if not None...')
ckpt_path = get_checkpoint(model_dir, load_ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
print('load checkpoint...')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
print(f"Model loaded from {ckpt_path}")
start_epoch = int(re.split(r'[._-]', load_ckpt_name)[1])
torch.set_rng_state(checkpoint['rng_state'])
torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
# model = load_model(model_dir, 'epoch-{}.pt'.format(max_epoch))

rst = {}

for t_user_label in new_test_user_labels:
    uid_list = user_bins[t_user_label]
    subset_test = {uid: users_test[uid] for uid in uid_list}                    # dict[uid] -> list
    subset_history = {uid: users_history_for_test[uid] for uid in uid_list}    # dict[uid] -> 1D tensor (seq[:-1])

    topK_list = [1, 5, 10]
    eval_dataset = EvalOneStepDataset(
        users_history_for_valid=subset_history,
        users_valid=users_valid,
        max_seq_len=getattr(args, "max_seq_len", 50),
        filter_seen=getattr(args, "filter_seen", True)
    )
    eval_dl = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    desc = f"Test {t_user_label:>7}"
    model.eval()
    iter_data = tqdm(eval_dl, total=len(eval_dl), ncols=100, desc=set_color(desc, 'pink'))

    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.tensor(np.arange(item_num) + 1, device=local_rank, dtype=torch.float32)

        for batch_idx, batch in enumerate(iter_data):
            hist_ids, labels = batch

            scores = model.predict(
                hist_ids=hist_ids,
                device=local_rank,
                item_word_embs=item_word_embs,
                is_use_modal=is_use_modal,
                topk=None,
                personalized=True,
                alpha_rel=getattr(args, "alpha_rel", 0.5)
            )  # [B, item_num(+1)]

            scores = scores[:, 1:]
            labels    = labels.to(local_rank)
            # Evaluate sample by sample (using your metrics_topKs)
            for label, score in zip(labels, scores):
                # metrics_topKs(score, label, item_rank, topK_list, device)
                eval_all_user.append(
                    metrics_topKs(score, label, item_rank, topK_list, local_rank)
                )

        eval_all_user = torch.stack(eval_all_user, dim=0)  # [num_users, 2, len(K)]
        mean_hitK = eval_all_user[:, 0, :].mean(dim=0).tolist()
        mean_ndcgK = eval_all_user[:, 1, :].mean(dim=0).tolist()
        mean_eval = mean_hitK + mean_ndcgK

        print("methods   {}".format('\t'.join([f'Hit@{k}' for k in topK_list] + [f'nDCG@{k}' for k in topK_list])))
        print_metrics_multi(mean_eval, 'test', topK_list)
        print(f"{t_user_label} Hit@10: {mean_eval[topK_list.index(10)]:.4f}")
        rst[t_user_label] = mean_eval

# Record model results to Excel file
record_model_result('Amazon_Prime_Pantry', model_name, rst, note=None)




