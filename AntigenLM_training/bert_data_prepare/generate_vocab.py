import csv
import sys
from collections import Counter

def read_fasta(file_path):
    """
    读取FASTA文件，返回所有蛋白质序列的列表。
    """
    sequences = []
    with open(file_path, 'r') as f:
        seq = ""
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 如果是header行，则保存之前的序列（如果有）
            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                    seq = ""
            else:
                seq += line
        if seq:
            sequences.append(seq)
    return sequences

def generate_ngrams(seq, n):
    """
    给定一个序列，生成所有长度为n的n-gram子串。
    """
    return [seq[i:i+n] for i in range(len(seq) - n + 1)]

def generate_vocab(sequences, ngram_sizes=(2, 3)):
    """
    对所有序列生成指定n-gram（默认2和3）的统计词汇表，
    返回一个Counter对象，其中键为n-gram，值为出现频率。
    """
    counter = Counter()
    for seq in sequences:
        for n in ngram_sizes:
            ngrams = generate_ngrams(seq, n)
            counter.update(ngrams)
    return counter

def save_vocab_to_csv(counter, output_file):
    """
    将词汇表保存为CSV文件，包含以下三列：
    - index: 根据频率排序后的索引
    - token: n-gram token
    - frequency: 该token的出现次数
    """
    # 按照出现频率从高到低排序
    tokens = counter.most_common()
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['index', 'token', 'frequency']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (token, freq) in enumerate(tokens):
            writer.writerow({'index': idx, 'token': token, 'frequency': freq})
 

if __name__ == "__main__":
    fasta_file = '/data0/chenkai/data/microLM-main/dataset/train_data_microLM.fasta'
    output_csv = '/data0/chenkai/data/microLM-main/Code/bert_data_prepare/microorganism-2-3.csv'
    
    print(f"Reading sequences from {fasta_file}...")
    sequences = read_fasta(fasta_file)
    print(f"Total sequences read: {len(sequences)}")
    
    print("Generating vocabulary for n-grams (length 2 and 3)...")
    vocab_counter = generate_vocab(sequences, ngram_sizes=(2, 3))
    print(f"Total unique tokens: {len(vocab_counter)}")
    
    print(f"Saving vocabulary to {output_csv}...")
    save_vocab_to_csv(vocab_counter, output_csv)
    print("Vocabulary generation completed.")
