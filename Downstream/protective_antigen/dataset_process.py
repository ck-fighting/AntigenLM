from torch.utils.data import Dataset
from Bio import SeqIO
import torch

class EmbDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    def __len__(self): 
        return len(self.labels)
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class antigenDataset(Dataset):
    def __init__(self, fasta_label_pairs):
        """
        fasta_label_pairs: List of tuples like [(fasta_path1, label1), (fasta_path2, label2)]
        """
        self.all_ids = []
        self.all_sequences = []
        self.all_labels = []
        
        for fasta_path, label in fasta_label_pairs:
            records = list(SeqIO.parse(fasta_path, "fasta"))
            for rec in records:
                self.all_ids.append(rec.description.strip())
                self.all_sequences.append(str(rec.seq))
                self.all_labels.append(label)

    def get_data(self):
        return self.all_ids, self.all_sequences, self.all_labels

#T4SE独立测试集
class T4SEtestProteinDataset(Dataset):
    def __init__(self, fasta_path, label=None, auto_parse_label=False):
        self.sequences = []
        self.ids = []
        self.labels = []
        records = list(SeqIO.parse(fasta_path, "fasta"))
        for rec in records:
            self.ids.append(rec.id)
            self.sequences.append(str(rec.seq))
            if auto_parse_label:
                try:
                    parsed_label = int(rec.id.split("|")[1])
                except:
                    parsed_label = -1
                self.labels.append(parsed_label)
            else:
                self.labels.append(label if label is not None else -1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.ids[idx], self.sequences[idx], self.labels[idx]
    
    
    
class fungiDataset:
    def __init__(self, dataframe):
        self.sequences = dataframe["sequence"].tolist()
        self.labels = dataframe["label"].tolist()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        return seq, label
    
    def get_all(self):
        return self.sequences, self.labels
    

class SARSExpDataset(Dataset):
    def __init__(self, dataframe):
        self.sequences = dataframe["sequence"].tolist()
        # 二值化标签：大于1设为1，否则设为0
        self.labels = [1 if label > 1 else 0 for label in dataframe["label"]]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        return seq, torch.tensor(label, dtype=torch.long)

    def get_all(self):
        return self.sequences, self.labels
