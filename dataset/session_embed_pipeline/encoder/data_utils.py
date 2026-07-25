import pandas as pd
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp


class BipartiteGraphDataset(Dataset):
    def __init__(self, args):
        super(BipartiteGraphDataset, self).__init__()
        self.device = args.device
        self.dataset = args.dataset
        self.neg_num = args.neg_num

        self.trainData, self.testData = [], []
        self.allUPos = []
        self.users, self.pos_items, self.neg_items = None, None, None

        with open('../datasets/general/' + self.dataset + '/train.txt', 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                user, items = int(line[0]), [int(item) for item in line[1:]]
                self.allUPos.append(items)
                for item in items:
                    self.trainData.append([user, item])

        with open('../datasets/general/' + self.dataset + '/test.txt', 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                user, items = int(line[0]), [int(item) for item in line[1:]]
                for item in items:
                    self.testData.append([user, item])

        self.trainData, self.testData = np.array(self.trainData), np.array(self.testData)
        self.trainDataSize = len(self.trainData)
        self.n_user = max(self.trainData[:, 0].max(), self.testData[:, 0].max()) + 1
        self.m_item = max(self.trainData[:, 1].max(), self.testData[:, 1].max()) + 1

        self.testDict = self.__build_test()
        self.uLabel = None
        self.get_labels()
        self.U, self.I, self.tuLabel = self.trainData[:, 0], self.trainData[:, 1], None

    def negative_sampling(self, batch_size, neg_num):
        neg_items = np.random.randint(0, self.m_item, (batch_size, neg_num))
        neg_items = torch.LongTensor(neg_items)
        return neg_items

    def get_labels(self):
        u_label = []
        allUPos = self.allUPos
        for user in range(self.n_user):
            user_pos = allUPos[user]
            user_label = torch.zeros(self.m_item, dtype=torch.float)
            user_label[user_pos] = 1.
            user_label = 0.9 * user_label + (1.0 / self.m_item)
            u_label.append(user_label.tolist())
        self.uLabel = torch.FloatTensor(u_label)

    def init_bathes(self):
        np.random.shuffle(self.U)
        self.tuLabel = self.uLabel[self.U]

    def get_user_pos_items(self, users):
        posItems = []
        for user in users:
            posItems.append(self.allUPos[user])
        return posItems

    def __convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float64)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def get_sparse_graph(self):
        print("loading matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz('datasets/general/' + self.dataset + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time.time()
                adj_mat = sp.dok_matrix((self.n_user + self.m_item, self.n_user + self.m_item), dtype=np.float64)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_user, self.n_user:] = R
                adj_mat[self.n_user:, :self.n_user] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time.time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz('datasets/general/' + self.dataset + '/s_pre_adj_mat.npz', norm_adj)

            self.Graph = self.__convert_sp_mat_to_sp_tensor(norm_adj).coalesce().to(self.device)

        return self.Graph

    def __build_test(self):
        tdu = {}
        for user, item in self.testData:
            tdu[user] = tdu.get(user, [])
            tdu[user].append(item)
        return tdu

    def __getitem__(self, idx):
        return self.trainData[idx]

    def __len__(self):
        return self.trainDataSize


class SequentialDataset(Dataset):
    def __init__(self, args):
        super(SequentialDataset, self).__init__()
        self.device = args.device
        self.dataset = args.dataset
        self.maxlen = args.maxlen

        self.trainData, self.valData, self.testData = [], [], []
        self.n_user, self.m_item = 0, 0

        base_path = "../datasets/sequential/" + self.dataset + "/"

        def _read_split_file(path):
            split = {}
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split(" ")
                    if len(parts) < 2:
                        continue
                    user = int(parts[0]) - 1
                    items = [int(item) for item in parts[1:]]
                    split[user] = items
            return split

        train_split = _read_split_file(base_path + "train.txt")
        val_split = _read_split_file(base_path + "val.txt")
        test_split = _read_split_file(base_path + "test.txt")
        val_sample_split = _read_split_file(base_path + "val_sample.txt")
        test_sample_split = _read_split_file(base_path + "test_sample.txt")

        all_users = sorted(set(train_split.keys()) | set(test_split.keys()))
        if all_users:
            self.n_user = max(all_users) + 1
        else:
            self.n_user = 0

        self.trainData = [[] for _ in range(self.n_user)]
        self.valData = [[] for _ in range(self.n_user)]
        self.testData = [[] for _ in range(self.n_user)]
        self.valCandidates = {}
        self.testCandidates = {}

        for user in all_users:
            train_items = train_split.get(user, [])
            test_items = test_split.get(user, [])
            if len(train_items) < 1 or len(test_items) < 1:
                continue

            self.trainData[user] = train_items
            self.valData[user] = val_split.get(user, [])
            self.testData[user] = test_items
            self.testCandidates[user] = test_sample_split.get(user, [test_items[0]])
            if user in val_sample_split and len(val_sample_split[user]) > 0:
                self.valCandidates[user] = val_sample_split[user]

            local_max = max(
                max(train_items),
                max(test_items),
                max(self.testCandidates[user]),
                max(self.valData[user]) if len(self.valData[user]) > 0 else 0,
                max(self.valCandidates[user]) if user in self.valCandidates else 0,
            )
            self.m_item = max(self.m_item, local_max)

        self.m_item = self.m_item + 1 if self.m_item > 0 else 1

    def negative_sampling(self, left, right, ts):
        t = np.random.randint(left, right)
        while t in ts:
            t = np.random.randint(left, right)
        return t

    def get_eval_users(self, subset='test'):
        if subset == 'val':
            return np.array(sorted(self.valCandidates.keys()))
        return np.array(sorted(self.testCandidates.keys()))

    def get_user_pos_items(self, users, subset='test'):
        posItems = []
        for user in users:
            if subset == 'val':
                posItems.append(self.valCandidates[user])
            else:
                posItems.append(self.testCandidates[user])
        return posItems

    def test_seq_generate(self, idx, subset='test'):
        seqs = []
        for i in idx:
            seq = np.zeros([self.maxlen], dtype=np.int32)
            if subset == 'test':
                context = self.trainData[i] + self.valData[i]
            else:
                context = self.trainData[i]

            context = context[-self.maxlen:]
            start = self.maxlen - len(context)
            seq[start:] = np.array(context, dtype=np.int32)
            seqs.append(seq)
        seqs = np.array(seqs)
        return seqs

    def __getitem__(self, idx):
        seq, pos, neg = (np.zeros([self.maxlen], dtype=np.int32),
                         np.zeros([self.maxlen], dtype=np.int32),
                         np.zeros([self.maxlen], dtype=np.int32))
        if len(self.trainData[idx]) == 0:
            return seq, pos, neg

        nxt = self.trainData[idx][-1]
        tmp = self.maxlen - 1
        ts = set(self.trainData[idx])
        for t in reversed(self.trainData[idx][:-1]):
            seq[tmp] = t
            pos[tmp] = nxt
            if nxt != 0:
                neg[tmp] = self.negative_sampling(1, self.m_item, ts)
            nxt = t
            tmp -= 1
            if tmp == -1:
                break
        return seq, pos, neg

    def __len__(self):
        return len(self.trainData)



