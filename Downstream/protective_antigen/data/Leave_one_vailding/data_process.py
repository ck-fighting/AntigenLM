import os
import pandas as pd
from Bio import SeqIO

# 文件夹路径
pos_dir = "/data0/chenkai/data/microLM-main/downstream/Data/protective_antigen_protective/positive3"
neg_dir = "/data0/chenkai/data/microLM-main/downstream/Data/protective_antigen_protective/negative_selected"

# 输出目录
out_dir = "/data0/chenkai/data/microLM-main/downstream/Data/protective_antigen_bacteria_overall/Leave_one_vailding"
os.makedirs(out_dir, exist_ok=True)

def fasta_to_df(fasta_path, label, species):
    """读取fasta并转成DataFrame"""
    records = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        records.append([rec.id, str(rec.seq), label, species])
    return pd.DataFrame(records, columns=["id", "sequence", "label", "species"])

# 统计物种（文件名前缀，去掉_selected/fasta后缀）
species_list = [f.replace("_selected.fasta", "") for f in os.listdir(neg_dir) if f.endswith(".fasta")]

# 遍历每个物种做留一实验
for sp in species_list:
    print(f"Processing {sp} ...")

    # 正负样本路径
    pos_file = os.path.join(pos_dir, sp + ".fasta")
    neg_file = os.path.join(neg_dir, sp + "_selected.fasta")

    # 转成DataFrame
    pos_df = fasta_to_df(pos_file, 1, sp)
    neg_df = fasta_to_df(neg_file, 0, sp)

    # 当前物种数据
    test_df = pd.concat([pos_df, neg_df], ignore_index=True)

    # 其他物种数据
    train_dfs = []
    for other_sp in species_list:
        if other_sp == sp:
            continue
        pos_file_o = os.path.join(pos_dir, other_sp + ".fasta")
        neg_file_o = os.path.join(neg_dir, other_sp + "_selected.fasta")
        if os.path.exists(pos_file_o):
            train_dfs.append(fasta_to_df(pos_file_o, 1, other_sp))
        if os.path.exists(neg_file_o):
            train_dfs.append(fasta_to_df(neg_file_o, 0, other_sp))
    train_df = pd.concat(train_dfs, ignore_index=True)

    # 保存CSV
    train_df.to_csv(os.path.join(out_dir, f"{sp}_train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, f"{sp}_test.csv"), index=False)
