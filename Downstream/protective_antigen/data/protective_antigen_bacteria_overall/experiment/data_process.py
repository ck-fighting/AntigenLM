# # import pandas as pd

# # # 读取原始 CSV
# # df = pd.read_csv("/data0/chenkai/data/microLM-main/downstream/Data/protective_antigen_bacteria_overall/experiment/experiment.csv")

# # # 去掉 sequence 列中的换行符和多余空格
# # df["sequence"] = df["sequence"].astype(str).str.replace(r"\s+", "", regex=True)

# # # 保存为新文件
# # df.to_csv("experiment_vaccine_antigens.csv", index=False)

# # print("处理完成，结果保存为 TB_vaccine_antigens_clean.csv")
# import pandas as pd
# import numpy as np
# import random

# # 所有氨基酸字母
# AA = list("ACDEFGHIKLMNPQRSTVWY")

# def corrupt_random_substitution(seq, p=0.05):
#     """以概率 p 把氨基酸随机替换为另一种"""
#     out = []
#     for a in seq:
#         if random.random() < p and a in AA:
#             choices = [x for x in AA if x != a]
#             out.append(random.choice(choices))
#         else:
#             out.append(a)
#     return ''.join(out)

# # 读取原始 csv
# df = pd.read_csv("/data0/chenkai/data/microLM-main/downstream/Data/protective_antigen_bacteria_overall/experiment/experiment_vaccine_antigens.csv")

# # 扰动概率
# ps = [0.05, 0.10, 0.15, 0.20]

# for p in ps:
#     df_copy = df.copy()
#     df_copy["sequence"] = df_copy["sequence"].astype(str).apply(lambda s: corrupt_random_substitution(s, p=p))
#     out_file = f"sequences_corrupted_{int(p*100)}.csv"
#     df_copy.to_csv(out_file, index=False, encoding="utf-8")
#     print(f"已保存 {out_file}")

import pandas as pd

# 不同扰动对应的文件
files = {
    "0%": "/data0/chenkai/data/microLM-main/downstream/Result/protective_antigen_bacteria/experiment/experiment_vaccine_antigens.csv",
    "5%": "/data0/chenkai/data/microLM-main/downstream/Result/protective_antigen_bacteria/experiment/experiment_vaccine_antigens_5.csv",
    "10%": "/data0/chenkai/data/microLM-main/downstream/Result/protective_antigen_bacteria/experiment/experiment_vaccine_antigens_10.csv",
    "15%": "/data0/chenkai/data/microLM-main/downstream/Result/protective_antigen_bacteria/experiment/experiment_vaccine_antigens_15.csv",
    "20%": "/data0/chenkai/data/microLM-main/downstream/Result/protective_antigen_bacteria/experiment/experiment_vaccine_antigens_20.csv",
}

results = []

for level, file in files.items():
    df = pd.read_csv(file)
    df["y_score"] = pd.to_numeric(df["y_score"], errors="coerce")
    grouped = df.groupby("Group")["y_score"].mean().reset_index()
    grouped["noise_level"] = level
    results.append(grouped)

# 合并结果
final_df = pd.concat(results)

# pivot 更直观：行=Group，列=扰动程度，值=平均分数
pivot_df = final_df.pivot(index="Group", columns="noise_level", values="y_score")

print(pivot_df)

# 保存
pivot_df.to_csv("avg_y_score_by_group.csv")

