import torch
import pandas as pd
from transformers import AutoTokenizer
from bert_finetuning_pathogen_dataset_second import StructureAwareDataCollatorForLanguageModeling  # 你的collator定义
from tqdm import tqdm
base_model_path = "/data0/chenkai/data/microLM-main/Result_microLM_80M/checkpoints/BERT-Pretrain-common-microLM/0325_151454_rank0"
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
# 加载你的模型使用的 tokenizer
# 读取 CSV 数据（需要包含 'sequence' 和 'structure' 两列）
df = pd.read_csv("/data0/chenkai/data/microLM-main/dataset/pathogen_second/pathogen_seq_ss.csv")  # 替换为你的路径
df = df.head(100)  # 只取前100条样本

# 构建原始样本格式
examples = []
max_len = 512

for i, row in df.iterrows():
    sequence = row["sequence"]
    ss = row["second_structure"]
    assert len(sequence) == len(ss), f"第{i}条数据长度不一致"

    # 分词处理
    spaced_seq = " ".join(list(sequence))
    tokenized = tokenizer(spaced_seq, padding="max_length", truncation=True,
                          max_length=max_len, return_tensors="pt")

    examples.append({
        "input_ids": tokenized["input_ids"][0],
        "attention_mask": tokenized["attention_mask"][0],
        "ss": ss[:max_len]
    })

# 创建结构感知数据整理器
collator = StructureAwareDataCollatorForLanguageModeling(tokenizer)

# 执行掩码
batch = collator(examples)
masked_ids = batch["input_ids"]
labels = batch["labels"]

# 打印前3个样本的掩码情况（也可以调整）
structure_stats = []

for i in range(len(examples)):
    ss = examples[i]["ss"][:max_len]
    label_ids = batch["labels"][i]  # Tensor
    mask_flags = label_ids != -100  # 被掩码的标志位

    # 初始化计数器
    total_C = ss.count("C")
    total_H = ss.count("H")
    total_E = ss.count("E")

    mask_C = mask_H = mask_E = 0

    for pos, tag in enumerate(ss):
        if pos >= len(mask_flags): break  # 越界保护
        if mask_flags[pos]:
            if tag == "C": mask_C += 1
            elif tag == "H": mask_H += 1
            elif tag == "E": mask_E += 1

    structure_stats.append({
        "seq_id": i,
        "total_C": total_C,
        "total_H": total_H,
        "total_E": total_E,
        "masked_C": mask_C,
        "masked_H": mask_H,
        "masked_E": mask_E
    })

# ✅ 打印前几条样本的统计结果
for stat in structure_stats[:5]:
    print(f"\n样本 {stat['seq_id']}：")
    print(f"  ➤ C: {stat['masked_C']}/{stat['total_C']} 掩码率={stat['masked_C']/stat['total_C']*100:.1f}%")
    print(f"  ➤ H: {stat['masked_H']}/{stat['total_H']} 掩码率={stat['masked_H']/stat['total_H']*100:.1f}%")
    print(f"  ➤ E: {stat['masked_E']}/{stat['total_E']} 掩码率={stat['masked_E']/stat['total_E']*100:.1f}%")

