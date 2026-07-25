import argparse
import pickle
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from eval_utils import RecallPrecision_atK, NDCG_atK, getLabel
from data_utils import SequentialDataset
from models import SASRec, BERT4Rec, GRU4Rec


MODEL = {
    'SASRec':   SASRec,
    'BERT4Rec': BERT4Rec,
    'GRU4Rec':  GRU4Rec,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',          type=str,   required=True, default='SASRec')
    parser.add_argument('--dataset',        type=str,   required=True, default='Beauty')
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--lr',             type=float, default=0.001)
    parser.add_argument('--batch_size',     type=int,   default=128)
    parser.add_argument('--dropout',        type=float, default=0.5)
    parser.add_argument('--cuda',           type=int,   default=0)
    parser.add_argument('--epochs',         type=int,   default=300)
    parser.add_argument('--weight_decay',   type=float, default=1e-4)
    parser.add_argument('--layers',         type=int,   default=2)
    parser.add_argument('--dim',            type=int,   default=64)
    parser.add_argument('--maxlen',         type=int,   default=50)
    parser.add_argument('--num_heads',      type=int,   default=1,    help='attention heads (BERT4Rec)')
    parser.add_argument('--mask_prob',      type=float, default=0.15, help='cloze mask probability (BERT4Rec)')
    parser.add_argument('--topk',           type=int,   nargs='+',    default=[1, 5, 10])
    parser.add_argument('--eval_freq',      type=int,   default=5)
    parser.add_argument('--lr_reduce_freq', type=int,   default=100)
    parser.add_argument('--gamma',          type=float, default=0.5)
    parser.add_argument('--save',           type=int,   default=0)
    return parser.parse_args()


# ── model-specific training steps ────────────────────────────────────────────

def _train_step_sasrec(model, batch, args):
    seq, pos, neg = batch
    indices = np.where(pos.numpy() != 0)
    seq  = seq.to(args.device)
    pos  = pos.to(args.device)
    neg  = neg.to(args.device)
    pos_labels = torch.ones(pos.shape, device=args.device)
    neg_labels = torch.zeros(neg.shape, device=args.device)
    pos_logits, neg_logits = model.loss_func(seq, pos, neg)
    loss = model.bce_loss(pos_logits[indices], pos_labels[indices])
    loss += model.bce_loss(neg_logits[indices], neg_labels[indices])
    return loss


def _train_step_bert4rec(model, batch, args):
    seq, _, _ = batch
    seq = seq.to(args.device)
    B, T = seq.shape
    rand = torch.rand(B, T, device=args.device)
    mask = (rand < args.mask_prob) & (seq != 0)
    masked_seq = seq.clone()
    masked_seq[mask] = model.mask_token
    labels = torch.where(mask, seq, torch.zeros_like(seq))
    return model.loss_func(masked_seq, labels)


def _train_step_gru4rec(model, batch, args):
    seq, pos, neg = batch
    seq = seq.to(args.device)
    lengths = (seq != 0).sum(dim=1).clamp(min=1) - 1
    last_pos = pos[torch.arange(len(pos)), lengths].to(args.device)
    last_neg = neg[torch.arange(len(neg)), lengths].to(args.device)
    return model.loss_func(seq, last_pos, last_neg)


TRAIN_STEP = {
    'SASRec':   _train_step_sasrec,
    'BERT4Rec': _train_step_bert4rec,
    'GRU4Rec':  _train_step_gru4rec,
}


# ── train / evaluate loops ────────────────────────────────────────────────────

def train(model, data_loader, optimizer, args):
    model.train()
    t = time.time()
    avg_loss = 0.0
    step = TRAIN_STEP[args.model]

    for i, batch in enumerate(data_loader):
        optimizer.zero_grad()
        loss = step(model, batch, args)
        loss.backward()
        optimizer.step()
        avg_loss += loss.cpu().item()

    avg_loss /= i + 1
    print(f'Average loss: {avg_loss:.4f}  |  Epoch time: {time.time() - t:.1f}s')


def evaluate(model, dataset, args, subset='test'):
    model.eval()
    with torch.no_grad():
        users = dataset.get_eval_users(subset=subset)
        if len(users) == 0:
            print(f'No users available for {subset} evaluation.')
            return 0.0

        results = {
            'Recall': np.zeros(len(args.topk)),
            'NDCG':   np.zeros(len(args.topk)),
        }
        batch_num = len(users) // args.batch_size + 1
        for i in range(batch_num):
            lo, hi = i * args.batch_size, (i + 1) * args.batch_size
            batch_users = users[lo:hi] if hi <= len(users) else users[lo:]
            if len(batch_users) == 0:
                continue

            selected_items = dataset.get_user_pos_items(batch_users, subset=subset)
            groundTruth = [[0]] * len(selected_items)

            seqs = dataset.test_seq_generate(batch_users, subset=subset)
            seqs = torch.LongTensor(seqs).to(args.device)

            ratings = model(seqs)
            candidate_tensor = torch.LongTensor(selected_items).to(args.device)
            ratings = torch.gather(ratings, 1, candidate_tensor)
            max_k = min(args.topk[-1], ratings.size(1))
            _, ratings_K = torch.topk(ratings, k=max_k)
            ratings_K = ratings_K.cpu().numpy()

            r = getLabel(groundTruth, ratings_K)
            for j, k in enumerate(args.topk):
                eval_k = min(k, max_k)
                _, rec = RecallPrecision_atK(groundTruth, r, eval_k)
                ndcg = NDCG_atK(groundTruth, r, eval_k)
                results['Recall'][j] += rec
                results['NDCG'][j]   += ndcg

        for key in results:
            results[key] /= float(len(users))
        print(f'{subset.upper()} evaluation:')
        for j, k in enumerate(args.topk):
            print(f'  Recall@{k}: {results["Recall"][j]:.4f}  NDCG@{k}: {results["NDCG"][j]:.4f}')
        return results['Recall'][-1]


# ── main ──────────────────────────────────────────────────────────────────────

args = parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
args.device = f'cuda:{args.cuda}' if args.cuda >= 0 else 'cpu'
torch.autograd.set_detect_anomaly(True)

dataset    = SequentialDataset(args)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
model      = MODEL[args.model](args, dataset)
print(model)
print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')

model = model.to(args.device)
for param in model.parameters():
    try:
        torch.nn.init.xavier_normal_(param.data)
    except Exception:
        pass

optimizer     = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
lr_scheduler  = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=args.lr_reduce_freq, gamma=args.gamma
)

t_total    = time.time()
best_recall = 0.0
has_val     = len(dataset.get_eval_users(subset='val')) > 0

for epoch in range(args.epochs):
    print(f'\nEpoch {epoch}')
    train(model, data_loader, optimizer, args)
    lr_scheduler.step()
    torch.cuda.empty_cache()

    if (epoch + 1) % args.eval_freq == 0:
        val_recall  = evaluate(model, dataset, args, subset='val') if has_val else 0.0
        test_recall = evaluate(model, dataset, args, subset='test')
        metric      = val_recall if has_val else test_recall

        if metric > best_recall:
            best_recall = metric
            if args.save == 1:
                base  = f'../datasets/sequential/{args.dataset}/{args.model}'
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, base + '.pth')
                pickle.dump(
                    model.embedding.weight.detach(),
                    open(base + '_item_embed.pkl', 'wb'),
                )
        torch.cuda.empty_cache()

print(f'\nTraining finished. Total time: {time.time() - t_total:.1f}s')
