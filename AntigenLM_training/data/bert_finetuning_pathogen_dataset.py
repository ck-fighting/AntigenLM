import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import math
import random


def min_power_greater_than(value, base=2):
    return base ** math.ceil(math.log(value, base))


class PathogenLoRADataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length, logger=None,
                 round_len=True, train_ratio=0.99, mode='train'):
        """
        train_ratio: 训练集所占比，例如 0.9 表示 90% 训练、10% 测试。
        mode: 'train' 或 'test'
        """
        self.logger = logger
        self.tokenizer = tokenizer
        # 取整至 2 的次幂，若 round_len=True
        self.max_len = int(min_power_greater_than(max_length)) if round_len else max_length
        self.mode = mode
        self.seqs = sequences

        # 预处理、切分蛋白质序列
        self.processed_seq = self._split_protein_sequences()
        self._has_logged_example = False

        if logger:
            logger.info(f"[{mode.upper()}] max_len used: {self.max_len}")

        # 数据划分（这里只分训练集与测试集）
        train_split = int(len(self.processed_seq) * train_ratio)
        if mode == 'train':
            self.sequences = self.processed_seq[:train_split]
        else:  # mode == 'test'
            self.sequences = self.processed_seq[train_split:]

        if logger:
            logger.info(f"[{mode.upper()}] Final subsequence: {len(self.sequences)}")

    def _split_protein_sequences(self):
        processed_seqs = []
        step = self.max_len // 2
        for seq in self.seqs:
            if len(seq) <= self.max_len:
                processed_seqs.append(seq)
            else:
                for i in range(0, len(seq) - self.max_len + 1, step):
                    processed_seqs.append(seq[i:i + self.max_len])
                if (len(seq) - self.max_len) % step != 0:
                    processed_seqs.append(seq[-self.max_len:])
        if self.logger:
            self.logger.info(f"Number of subsequences after splitting: {len(processed_seqs)}")
        return processed_seqs

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        tokenized_input = self.tokenizer.encode(
            self._insert_whitespace(list(seq)),
            truncation=True, max_length=self.max_len
        )
        if not self._has_logged_example:
            # 可以在这里打印一些调试信息
            self._has_logged_example = True
        return {"input_ids": torch.tensor(tokenized_input, dtype=torch.long)}

    def _insert_whitespace(self, token_list):
        """
        在每个 token 之间插入空格，用于后续分词编码。
        """
        return " ".join(token_list)


class ProteinDataLoader:
    def __init__(self, data_path, tokenizer, batch_size, max_length,
                 seed=42, train_ratio=0.99, logger=None):
        """
        只保留 train/test，不再单独创建 valid。
        train_ratio: 训练集所占比，例如 0.9 表示 90% 训练、10% 测试。
        """
        self.logger = logger
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.seed = seed
        self.train_ratio = train_ratio

        # 加载序列
        self.sequences = self._load_sequences(data_path)
        random.shuffle(self.sequences)

        # 构造数据集（仅 train 和 test）
        self.train_dataset = PathogenLoRADataset(
            self.sequences, tokenizer, max_length=max_length,
            logger=logger, train_ratio=train_ratio, mode='train'
        )
        self.test_dataset = PathogenLoRADataset(
            self.sequences, tokenizer, max_length=max_length,
            logger=logger, train_ratio=train_ratio, mode='test'
        )

        self.train_sampler = RandomSampler(self.train_dataset)
        self.test_sampler = SequentialSampler(self.test_dataset)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
                                       sampler=self.train_sampler)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size,
                                      sampler=self.test_sampler)

    def _load_sequences(self, path):
        sequences = []
        if path.endswith('.txt') or path.endswith('.fasta'):
            with open(path, 'r') as f:
                seq = ""
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if seq:
                            sequences.append(seq)
                            seq = ""
                    else:
                        seq += line
                if seq:
                    sequences.append(seq)
        elif path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(path)
            sequences = df['sequence'].tolist()
        else:
            raise ValueError("Unsupported file type. Use .fasta, .txt, or .csv")

        if self.logger:
            self.logger.info(f"Loaded {len(sequences)} sequences from {path}")
        return sequences

    def __iter__(self):
        """默认返回训练集迭代器，可根据需要修改。"""
        return iter(self.train_loader)

    def __len__(self):
        """返回训练集的批次数。"""
        return len(self.train_loader)

    @property
    def sampler(self):
        return self.train_sampler

    def split_dataset(self, test=False):
        """
        获取 train_loader 或 test_loader。
        """
        if test:
            return self.test_loader
        return self.train_loader

    def get_token_list(self):
        return self.tokenizer.convert_ids_to_tokens(range(self.tokenizer.vocab_size))

    def get_dataset(self, mode='train'):
        """
        获取对应的 Dataset。
        """
        if mode == 'train':
            return self.train_dataset
        elif mode == 'test':
            return self.test_dataset
        else:
            raise ValueError(f"Unsupported mode: {mode}")
