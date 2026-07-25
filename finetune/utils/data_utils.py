import pandas as pd
import numpy as np
import torch
import time
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader


class BipartiteGraphDataset(Dataset):
    def __init__(self, dataset):
        super(BipartiteGraphDataset, self).__init__()
        self.dataset = dataset

        self.trainData, self.allPos, self.testData = [], {}, {}
        self.n_user, self.m_item = 0, 0
        with open(self.dataset + 'train.txt', 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                user, items = int(line[0]), [int(item) + 1 for item in line[1:]]
                self.allPos[user] = items
                for item in items:
                    self.trainData.append([user, item])
                self.n_user = max(self.n_user, user)
                self.m_item = max(self.m_item, max(items))

        with open(self.dataset + 'test.txt', 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                user, items = int(line[0]), [int(item) + 1 for item in line[1:]]
                self.testData[user] = items
                self.n_user = max(self.n_user, user)
                self.m_item = max(self.m_item, max(items))

        self.n_user, self.m_item = self.n_user + 1, self.m_item + 1

    def __getitem__(self, idx):
        user, item = self.trainData[idx]
        return user, self.allPos[user], item

    def __len__(self):
        return len(self.trainData)


@dataclass
class BipartiteGraphCollator:
    def __call__(self, batch) -> dict:
        user, items, labels = zip(*batch)
        bs = len(user)
        max_len = max([len(item) for item in items])
        inputs = [[user[i]] + items[i] + [0] * (max_len - len(items[i])) for i in range(bs)]
        inputs_mask = [[1] + [1] * len(items[i]) + [0] * (max_len - len(items[i])) for i in range(bs)]
        labels = [[label] for label in labels]
        inputs, inputs_mask, labels = torch.LongTensor(inputs), torch.LongTensor(inputs_mask), torch.LongTensor(labels)

        return {
            "inputs": inputs,
            "inputs_mask": inputs_mask,
            "labels": labels
        }


class SequentialDataset(Dataset):
    def __init__(self, dataset, maxlen):
        super(SequentialDataset, self).__init__()
        self.dataset = dataset
        self.maxlen = maxlen

        self.trainData, self.valData, self.testData = [], {}, {}
        self.n_user, self.m_item = 0, 0

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

        train_split = _read_split_file(self.dataset + "train.txt")
        test_split = _read_split_file(self.dataset + "test.txt")
        val_split = _read_split_file(self.dataset + "val.txt")
        test_sample_split = _read_split_file(self.dataset + "test_sample.txt")

        # Build a compact item remapping: raw item id -> contiguous index [0, n_unique).
        # Without this, m_item = max_raw_id+1 (~383k for LastFM) which makes the
        # score head and embedding table enormous even though only ~3k unique items exist.
        all_raw_ids = set()
        all_users = sorted(set(train_split.keys()) | set(test_split.keys()))
        for user in all_users:
            for item in train_split.get(user, []):
                all_raw_ids.add(item)
            for item in test_split.get(user, []):
                all_raw_ids.add(item)
            for item in val_split.get(user, []):
                all_raw_ids.add(item)
            for item in test_sample_split.get(user, []):
                all_raw_ids.add(item)
        # index 0 is reserved as the padding/unknown token
        self.item_map = {raw: idx + 1 for idx, raw in enumerate(sorted(all_raw_ids))}

        def remap(items):
            return [self.item_map[i] for i in items if i in self.item_map]

        self.allPos = {}
        for user in all_users:
            train_items = remap(train_split.get(user, []))
            test_items  = remap(test_split.get(user, []))
            if len(train_items) < 1 or len(test_items) < 1:
                continue

            sample_items = remap(test_sample_split.get(user, []))
            if not sample_items:
                sample_items = [test_items[0]]
            self.allPos[user] = sample_items

            self.testData[user] = [train_items, test_items[0]]
            val_items = remap(val_split.get(user, []))
            if val_items:
                self.valData[user] = [train_items, val_items[0]]
            else:
                self.valData[user] = []

            length = min(len(train_items), self.maxlen)
            for t in range(length):
                self.trainData.append([train_items[:-length + t], train_items[-length + t]])

            self.n_user = max(self.n_user, user)

        self.n_user  = self.n_user + 1
        self.m_item  = len(self.item_map) + 1  # +1 for the padding index 0

    def get_user_pos_items(self, users):
        posItems = []
        for user in users:
            posItems.append(self.allPos[user])
        return posItems

    def __getitem__(self, idx):
        seq, label = self.trainData[idx]
        return seq, label

    def __len__(self):
        return len(self.trainData)

@dataclass
class SequentialCollator:
    max_len: int = 50  # matches SequentialDataset maxlen; keeps item tokens ≤ this

    def __call__(self, batch) -> dict:
        seqs, labels = zip(*batch)
        # Truncate each sequence to the most recent max_len items (right-aligned).
        seqs = [seq[-self.max_len:] if len(seq) > self.max_len else seq for seq in seqs]
        pad_len = max(max(len(seq) for seq in seqs), 2)
        inputs = [[0] * (pad_len - len(seq)) + seq for seq in seqs]
        inputs_mask = [[0] * (pad_len - len(seq)) + [1] * len(seq) for seq in seqs]
        labels = [[label] for label in labels]
        inputs, inputs_mask, labels = torch.LongTensor(inputs), torch.LongTensor(inputs_mask), torch.LongTensor(labels)

        return {
            "inputs": inputs,
            "inputs_mask": inputs_mask,
            "labels": labels
        }

