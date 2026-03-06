import math
import os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from bert_data_prepare.tokenizer import get_tokenizer

def min_power_greater_than(value, base=2):
    """
    Return the smallest power of the given base that is greater than or equal to the value.
    """
    return base ** math.ceil(math.log(value, base))

class SelfSupervisedDataset(Dataset):
    def __init__(self, seqs, split_fun, tokenizer, max_len, logger, round_len=True):
        """
        Args:
            seqs (list): 原始蛋白质序列列表
            split_fun (callable): 分词函数，用于对序列进行分词
            tokenizer: BERT 分词器，用于编码序列
            max_len (int): 最大序列长度（窗口大小）
            logger: 日志对象，用于记录信息
            round_len (bool): 是否将 max_len 调整为大于或等于它的最小 2 的幂（默认 True）
        """
        self.logger = logger
        self.seqs = seqs
        self.split_fun = split_fun
        self.tokenizer = tokenizer
        
        self.logger.info(f"Creating dataset with {len(self.seqs)} sequences")
        self.max_len = int(min_power_greater_than(max_len, 2)) if round_len else max_len
        self.logger.info(f"Max sequence length set to: {self.max_len}")
        
        self.processed_seqs = self._split_protein_sequences()
        self._has_logged_example = False
        self.logger.info(f"Total processed sequences: {len(self.processed_seqs)}")
    
    def _split_protein_sequences(self):
        """
        使用滑动窗口方法切割蛋白质序列，并产生重叠区域。
        窗口大小为 self.max_len，步长默认为 self.max_len // 2。
        """
        processed_seqs = []
        step = self.max_len // 2
        
        for seq in self.seqs:
            seq_len = len(seq)
            if seq_len <= self.max_len:
                processed_seqs.append(seq)
            else:
                for i in range(0, seq_len - self.max_len + 1, step):
                    processed_seqs.append(seq[i:i + self.max_len])
                # 如果最后一段窗口未完全覆盖，则添加最后一个完整窗口
                if (seq_len - self.max_len) % step != 0:
                    processed_seqs.append(seq[-self.max_len:])
        
        self.logger.info(f"Number of subsequences after splitting: {len(processed_seqs)}")
        return processed_seqs

    def __len__(self):
        return len(self.processed_seqs)

    def __getitem__(self, index):
        seq = self.processed_seqs[index]
        tokenized_input = self.tokenizer.encode(
            self._insert_whitespace(self.split_fun(seq)),
            truncation=True, max_length=self.max_len
        )
        if not self._has_logged_example:
            self._has_logged_example = True
        return {"input_ids": torch.tensor(tokenized_input, dtype=torch.long)}

    def merge(self, other):
        """
        Merge this dataset with another SelfSupervisedDataset and return a new dataset.
        """
        merged_seqs = self.seqs + other.seqs
        self.logger.info(
            f"Merging datasets: sizes {len(self)} and {len(other)} resulting in {len(merged_seqs)} sequences"
        )
        return SelfSupervisedDataset(merged_seqs, self.split_fun, self.tokenizer, self.max_len, self.logger)

    def _insert_whitespace(self, token_list):
        """
        在每个 token 之间插入空格，用于后续分词编码。
        """
        return " ".join(token_list)

class MAADataset:
    def __init__(self, config, logger, seed, seq_dir, tokenizer_name, vocab_dir, token_length_list, seq_name, max_len=None, test_split=0.1):
        """
        Args:
            config: 配置对象，包含保存目录等信息
            logger: 日志对象
            seed (int): 随机种子
            seq_dir (str): 序列文件路径（FASTA 格式）
            tokenizer_name (str): 分词器名称
            vocab_dir (str): 词汇表所在目录
            token_length_list: 用于分词器初始化的 token 长度列表
            seq_name (str): 序列名称（备用参数）
            max_len (int, optional): 指定最大序列长度。若为 None，则计算所有序列的最大分词数
            test_split (float): 测试集比例
        """
        self.config = config
        self.logger = logger
        self.seed = seed
        self.test_split = test_split
        self.seq_list = self._load_seq(seq_dir)
        
        self.logger.info("Creating tokenizer...")
        self.tokenizer = get_tokenizer(tokenizer_name, False, logger, vocab_dir, token_length_list)
        self.split_fun = self.tokenizer.split

        computed_max_len = max_len or max(len(self.split_fun(s)) for s in self.seq_list)
        self.max_len = int(min_power_greater_than(computed_max_len, 2))
        
        self.bert_tokenizer = self.tokenizer.get_bert_tokenizer(max_len=self.max_len)
        # 确保保存目录存在
        os.makedirs(config._save_dir, exist_ok=True)
        if not os.listdir(config._save_dir):
            self.bert_tokenizer.save_pretrained(config._save_dir)

    def _load_seq(self, seq_dir):
        """
        从 FASTA 文件中加载序列。
        """
        seq_list = []
        with open(seq_dir, "r") as f:
            sequence = ""
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if sequence:
                        seq_list.append(sequence)
                        sequence = ""
                else:
                    sequence += line
            if sequence:
                seq_list.append(sequence)
        self.logger.info(f"Loaded {len(seq_list)} sequences from {seq_dir}.")
        return seq_list
    
    def get_token_list(self):
        return self.tokenizer.token_with_special_list

    def get_vocab_size(self):
        return len(self.tokenizer.token2index_dict)

    def get_pad_token_id(self):
        return self.tokenizer.token2index_dict[self.tokenizer.PAD]

    def get_tokenizer(self):
        return self.bert_tokenizer

    def _split(self):
        """
        将序列列表划分为训练集和测试集，
        如果存在多于一个唯一序列，则使用 stratify 参数保证比例平衡。
        """
        stratify = self.seq_list if len(set(self.seq_list)) > 1 else None
        return train_test_split(
            self.seq_list, test_size=self.test_split, random_state=self.seed, stratify=stratify
        )

    def get_dataset(self):
        dataset = SelfSupervisedDataset(self.seq_list, self.split_fun, self.bert_tokenizer, self.max_len, self.logger)
        self.logger.info(f"Final dataset size: {len(dataset)}")
        return dataset
