import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import math
import random
from transformers import DataCollatorForLanguageModeling


def min_power_greater_than(value, base=2):
    return base ** math.ceil(math.log(value, base))


class PathogenLoRADataset(Dataset):
    def __init__(self, sequences_with_structure, tokenizer, max_length, logger=None,
                 round_len=True, train_ratio=0.99, mode='train'):
        self.logger = logger
        self.tokenizer = tokenizer
        self.max_len = int(min_power_greater_than(max_length)) if round_len else max_length
        self.mode = mode
        self.seqs_with_ss = sequences_with_structure  # [(seq, ss), ...]

        self.processed = self._split_sequences_and_structure()
        self._has_logged_example = False

        train_split = int(len(self.processed) * train_ratio)
        if mode == 'train':
            self.data = self.processed[:train_split]
        else:
            self.data = self.processed[train_split:]

        if logger:
            logger.info(f"[{mode.upper()}] Final subsequence: {len(self.data)}")

    def _split_sequences_and_structure(self):
        results = []
        step = self.max_len // 2
        for seq, ss in self.seqs_with_ss:
            assert len(seq) == len(ss), "Sequence and structure lengths must match!"
            if len(seq) <= self.max_len:
                results.append((seq, ss))
            #滑动窗口加重叠
            else:
                for i in range(0, len(seq) - self.max_len + 1, step):
                    results.append((seq[i:i + self.max_len], ss[i:i + self.max_len]))
                if (len(seq) - self.max_len) % step != 0:
                    results.append((seq[-self.max_len:], ss[-self.max_len:]))
        return results

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq, ss = self.data[index]
        # 用 tokenizer 同时返回 input_ids 和 attention_mask
        encoding = self.tokenizer(
            self._insert_whitespace(list(seq)),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),         # Tensor(seq_len)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # Tensor(seq_len)
            "ss": ss  # 字符串，长度 >= seq_len
        }


    def _insert_whitespace(self, token_list):
        return " ".join(token_list)



class ProteinDataLoader:
    def __init__(self, data_path, tokenizer, batch_size, max_length,
                 seed=42, train_ratio=0.99, logger=None,**kwargs):
        """
        支持 CSV 格式，包含 'sequence' 和 'structure' 两列。
        """
        self.logger = logger
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.seed = seed
        self.train_ratio = train_ratio

        # 加载带二级结构的序列数据
        self.sequence_with_structure = self._load_sequences_with_structure(data_path)
        random.seed(seed)
        random.shuffle(self.sequence_with_structure)

        # 构造 train/test 数据集
        self.train_dataset = PathogenLoRADataset(
            self.sequence_with_structure, tokenizer, max_length=max_length,
            logger=logger, train_ratio=train_ratio, mode='train'
        )
        self.test_dataset = PathogenLoRADataset(
            self.sequence_with_structure, tokenizer, max_length=max_length,
            logger=logger, train_ratio=train_ratio, mode='test'
        )

        self.train_sampler = RandomSampler(self.train_dataset)
        self.test_sampler = SequentialSampler(self.test_dataset)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, sampler=self.train_sampler)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, sampler=self.test_sampler)

    def _load_sequences_with_structure(self, path):
        """
        读取 CSV 文件，要求包含 'sequence' 和 'structure' 两列。
        返回 [(seq, ss), ...]
        """
        if not path.endswith('.csv'):
            raise ValueError("Only CSV file with 'sequence' and 'structure' columns is supported.")

        import pandas as pd
        df = pd.read_csv(path)

        if 'sequence' not in df.columns or 'second_structure' not in df.columns:
            raise ValueError("CSV must contain 'sequence' and 'strsecond_structureucture' columns.")

        sequences = df['sequence'].tolist()
        structures = df['second_structure'].tolist()

        if self.logger:
            self.logger.info(f"Loaded {len(sequences)} sequence-structure pairs from {path}")

        return list(zip(sequences, structures))

    def __iter__(self):
        """默认返回训练集的迭代器"""
        return iter(self.train_loader)

    def __len__(self):
        """训练集中 batch 数量"""
        return len(self.train_loader)

    @property
    def sampler(self):
        return self.train_sampler

    def split_dataset(self, test=False):
        return self.test_loader if test else self.train_loader

    def get_token_list(self):
        return self.tokenizer.convert_ids_to_tokens(range(self.tokenizer.vocab_size))

    def get_dataset(self, mode='train'):
        if mode == 'train':
            return self.train_dataset
        elif mode == 'test':
            return self.test_dataset
        else:
            raise ValueError(f"Unsupported mode: {mode}")



