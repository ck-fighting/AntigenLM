# # import pandas as pd
# # import re
# # from pathlib import Path

# # # ========= 路径配置 =========
# # SRC_CSV = "/data0/chenkai/data/microLM-main/downstream/Code_antigen_MHC/UnifyImmun-main/data/data_HLA_mi/mhc_ligand_table_export_1756972988.csv"           # 原始 CSV
# # HLA_PSEUDO = "/data0/chenkai/data/microLM-main/downstream/Code_antigen_MHC/UnifyImmun-main/data/data_HLA_mi/hlaI_pseudo_seq.csv"   # HLA 伪序列/氨基酸序列表
# # OUT_CSV = "/data0/chenkai/data/microLM-main/downstream/Code_antigen_MHC/UnifyImmun-main/data/data_HLA_I_micro/extracted_with_hla_seq_Negative.csv"

# # # ========= 工具函数 =========
# # def excel_col_to_idx(col_letters: str) -> int:
# #     """把 Excel 列字母转换为 0-based 索引，如 'A'->0, 'L'->11, 'DD'->107"""
# #     col_letters = col_letters.strip().upper()
# #     n = 0
# #     for ch in col_letters:
# #         n = n * 26 + (ord(ch) - ord('A') + 1)
# #     return n - 1

# # def normalize_header(s: str) -> str:
# #     """标准化列名：去首尾空格，把换行替换为空格，并合并多空格"""
# #     return re.sub(r"\s+", " ", str(s).strip())

# # def normalize_allele(raw: str) -> str:
# #     """
# #     规范化单个 HLA 等位基因字符串：
# #       - 去空格、统一大写
# #       - 确保带 'HLA-' 前缀
# #       - 把六位/更多位编号截成四位（例如 *58:01:01 -> *58:01）
# #       - 兼容 A0201 / B5801 这类简写（尽量转为标准）
# #     """
# #     if not raw or pd.isna(raw):
# #         return ""
# #     s = str(raw).strip().upper()
# #     s = s.replace(" ", "")

# #     # 常见简写修复：A0201 -> HLA-A*02:01, B5801 -> HLA-B*58:01 等
# #     m = re.match(r'^[ABC]\d{4}$', s)  # A0201 / B5801 / C0702
# #     if m:
# #         locus = s[0]
# #         s = f"HLA-{locus}*{s[1:3]}:{s[3:5]}"

# #     # 如果没有 "HLA-" 前缀但像 A*02:01 这种
# #     if not s.startswith("HLA-") and re.match(r'^[ABC]\*\d{2}:\d{2}', s):
# #         s = "HLA-" + s

# #     # 常规标准：HLA-A*02:01:01 之类 → 截成四位
# #     m = re.match(r'^(HLA-[ABC]\*\d{2}:\d{2})', s)
# #     if m:
# #         return m.group(1)

# #     # 兼容如 HLA-B5801（少见）
# #     m = re.match(r'^HLA-([ABC])(\d{4})$', s)
# #     if m:
# #         locus, digits = m.groups()
# #         return f"HLA-{locus}*{digits[:2]}:{digits[2:]}"

# #     # 已经是四位的或其他情况就原样返回（后面再尝试直接查表）
# #     return s

# # def split_alleles(cell: str):
# #     """把一个单元格里的多个等位基因拆开；支持 ; , / | 和空白分隔"""
# #     if pd.isna(cell) or str(cell).strip() == "":
# #         return []
# #     parts = re.split(r'[;,\|/]+|\s+', str(cell).strip())
# #     parts = [p for p in parts if p]
# #     # 有些数据会把 'HLA-B*58:01;HLA-A*02:01' 写在一起，中间无空格；上面的分隔能处理
# #     return parts

# # # ========= 读取并准备 HLA 伪序列表 =========
# # hla_df = pd.read_csv(HLA_PSEUDO)
# # # 尝试猜测等位基因列与序列列
# # cols_lower = {c.lower(): c for c in hla_df.columns}
# # allele_col = next((cols_lower[c] for c in cols_lower if 'allele' in c or 'hla' in c), None)
# # seq_col = next((cols_lower[c] for c in cols_lower if 'pseudo' in c or 'seq' in c or 'aa' in c), None)
# # if allele_col is None or seq_col is None:
# #     raise ValueError(f"在 {HLA_PSEUDO} 里找不到等位基因列/序列列，请检查表头。现有列：{list(hla_df.columns)}")

# # # 建索引：既包含四位，也保留原样 → 提升命中率
# # hla_map = {}
# # for _, row in hla_df.iterrows():
# #     key_raw = str(row[allele_col]).strip()
# #     seq = str(row[seq_col]).strip()
# #     if not key_raw or not seq:
# #         continue
# #     hla_map[key_raw.upper().replace(" ", "")] = seq
# #     key_norm = normalize_allele(key_raw)
# #     if key_norm:
# #         hla_map[key_norm.upper().replace(" ", "")] = seq

# # # ========= 读取原始 CSV，定位 L 与 DD 列 =========
# # df = pd.read_csv(SRC_CSV, dtype=str)  # 全部按字符串读，避免类型干扰
# # # 标准化列名（处理换行）
# # df.columns = [normalize_header(c) for c in df.columns]

# # # 优先用 Excel 列号（L、DD），如失败再尝试列名匹配
# # try:
# #     L_idx = excel_col_to_idx("L")
# #     DD_idx = excel_col_to_idx("DD")
# #     col_L = df.columns[L_idx]
# #     col_DD = df.columns[DD_idx]
# # except Exception:
# #     # 退路：按标题名找 DD 列；L 列由你自己指定的标题（如果知道）
# #     # 题述中 DD 列标题是 "MHC Restriction\nName"（带换行），normalize 后应为：
# #     dd_name_candidates = {"MHC Restriction Name", "MHC Restriction", "Restriction Name"}
# #     found_dd = [c for c in df.columns if normalize_header(c) in dd_name_candidates]
# #     if not found_dd:
# #         raise ValueError("找不到 DD 列（MHC Restriction Name）。请确认文件或用字母列号定位。")
# #     col_DD = found_dd[0]
# #     # L 列如果没有固定标题，这里演示用 L 列号取
# #     L_idx = excel_col_to_idx("L")
# #     col_L = df.columns[L_idx]

# # # ========= 从两列构建新表，并根据 DD 列映射序列 =========
# # out = df[[col_L, col_DD]].copy()
# # out.rename(columns={col_L: "L_col", col_DD: "MHC_Restriction_Name"}, inplace=True)

# # unmatched = set()

# # def map_cell_to_seq(cell: str) -> str:
# #     alleles = split_alleles(cell)
# #     seqs = []
# #     for a in alleles:
# #         a_norm = normalize_allele(a)
# #         # 两种 key 都尝试
# #         keys = [
# #             a.upper().replace(" ", ""),
# #             a_norm.upper().replace(" ", "") if a_norm else ""
# #         ]
# #         hit = None
# #         for k in keys:
# #             if k and k in hla_map:
# #                 hit = hla_map[k]
# #                 break
# #         if hit is None:
# #             unmatched.add(a)
# #             seqs.append("")  # 占位；也可以选择跳过
# #         else:
# #             seqs.append(hit)
# #     # 用分号拼回；如果只有一个就返回一个
# #     return ";".join([s for s in seqs if s != ""]) if any(seqs) else ""

# # out["HLA_Sequence"] = out["MHC_Restriction_Name"].apply(map_cell_to_seq)

# # # ========= 保存结果 =========
# # Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
# # out.to_csv(OUT_CSV, index=False, encoding="utf-8")
# # print(f"已保存到: {OUT_CSV}")

# # if unmatched:
# #     print("\n以下等位基因未匹配到序列（请检查写法/是否为非 I 类等位基因/是否需要扩展库）：")
# #     for x in sorted(unmatched):
# #         print("  -", x)
        
        
        
        
        
# # import pandas as pd

# # # 读取刚才生成的文件
# # df = pd.read_csv("/data0/chenkai/data/microLM-main/downstream/Code_antigen_MHC/UnifyImmun-main/data/data_HLA_I_micro/extracted_with_hla_seq_Negative.csv")

# # # 去掉 HLA_Sequence 为空或空字符串的行
# # df = df[df["HLA_Sequence"].notna() & (df["HLA_Sequence"].str.strip() != "")]

# # # 新增一列 label，全设为 1
# # df["label"] = 0

# # # 保存
# # df.to_csv("extracted_with_hla_seq_labeled_negative.csv", index=False, encoding="utf-8")

# # print(f"处理完成，最终得到 {len(df)} 行，已保存到 extracted_with_hla_seq_labeled_negative.csv")


# # import pandas as pd
# # import re

# # # ===== 配置：把下面四个文件/列名改成你的实际名称 =====
# # A_CSV = "/data0/chenkai/data/microLM-main/downstream/Code_antigen_MHC/UnifyImmun-main/data/data_HLA_I_micro/extracted_with_hla_seq_labeled.csv"          # 第一个csv
# # B_CSV = "/data0/chenkai/data/microLM-main/downstream/Code_antigen_MHC/UnifyImmun-main/data/data_HLA/dataset.csv"          # 第二个csv
# # PEP_COL = "peptide"           # 两个文件里肽序列列名
# # HLA_SEQ_COL = "HLA"  # 两个文件里HLA序列列名
# # OUT_CSV = "peptide_hla_intersection.csv"

# # def normalize_pep(x: str) -> str:
# #     if pd.isna(x): return ""
# #     return str(x).strip().upper().replace(" ", "")

# # def split_multi(cell: str):
# #     """把一个单元格里的多个序列拆开；支持 ; , / | 和空白分隔"""
# #     if pd.isna(cell) or str(cell).strip() == "":
# #         return []
# #     parts = re.split(r'[;,\|/]+|\s+', str(cell).strip())
# #     return [p for p in parts if p]

# # def load_and_prepare(path: str) -> pd.DataFrame:
# #     df = pd.read_csv(path, dtype=str)
# #     # 规范 peptide
# #     df["peptide_norm"] = df[PEP_COL].apply(normalize_pep)
# #     # 拆分并展开 HLA_Sequence
# #     df = df.assign(HLA_split=df[HLA_SEQ_COL].apply(split_multi)).explode("HLA_split", ignore_index=True)
# #     # 清洗序列空白
# #     df["HLA_split"] = df["HLA_split"].fillna("").str.strip()
# #     # 仅保留非空
# #     df = df[(df["peptide_norm"] != "") & (df["HLA_split"] != "")]
# #     # 只保留匹配所需的两列
# #     return df[["peptide_norm", "HLA_split"]].drop_duplicates()

# # A = load_and_prepare(A_CSV)
# # B = load_and_prepare(B_CSV)

# # # 取交集（内连接）
# # inter = A.merge(B, on=["peptide_norm", "HLA_split"], how="inner").drop_duplicates()

# # # 恢复为原列名保存（若需要保留规范化版本，可改列名）
# # inter = inter.rename(columns={"peptide_norm": PEP_COL, "HLA_split": HLA_SEQ_COL})
# # inter.to_csv(OUT_CSV, index=False, encoding="utf-8")

# # print(f"完成！匹配到 {len(inter)} 行，已保存到 {OUT_CSV}")
# # import pandas as pd

# # # ========= 配置 =========
# # A_CSV = "/data0/chenkai/data/microLM-main/downstream/Code_antigen_MHC/UnifyImmun-main/data/data_HLA/dataset.csv"          # 第一个csv
# # B_CSV = "/data0/chenkai/data/microLM-main/downstream/Code_antigen_MHC/UnifyImmun-main/data/data_HLA_I_micro/extracted_with_hla_seq_labeled.csv"          # 第二个csv
# # PEP_COL = "peptide"           # peptide 列名
# # HLA_SEQ_COL = "HLA"  # HLA 序列列名
# # LABEL_COL = "label"           # 两个文件里的标签列名（假设相同）
# # OUT_CSV = "peptide_hla_intersection_with_labels.csv"

# # # 读取
# # A = pd.read_csv(A_CSV, dtype=str)
# # B = pd.read_csv(B_CSV, dtype=str)

# # # 预处理（统一大小写/空格）
# # A[PEP_COL] = A[PEP_COL].str.upper().str.strip()
# # B[PEP_COL] = B[PEP_COL].str.upper().str.strip()
# # A[HLA_SEQ_COL] = A[HLA_SEQ_COL].str.strip()
# # B[HLA_SEQ_COL] = B[HLA_SEQ_COL].str.strip()

# # # 重命名 label 列，避免冲突
# # A = A.rename(columns={LABEL_COL: "label_A"})
# # B = B.rename(columns={LABEL_COL: "label_B"})

# # # 取交集，并保留两个 label
# # inter = A.merge(
# #     B[[PEP_COL, HLA_SEQ_COL, "label_B"]],
# #     on=[PEP_COL, HLA_SEQ_COL],
# #     how="inner"
# # )

# # # 保存
# # inter.to_csv(OUT_CSV, index=False, encoding="utf-8")
# # print(f"完成！匹配到 {len(inter)} 行，结果已保存到 {OUT_CSV}")
# import pandas as pd

# # ===== 配置 =====
# CSV_FILE = "/data0/chenkai/data/microLM-main/downstream/Code_antigen_MHC/UnifyImmun-main/data/data_HLA_I_micro/extracted_with_hla_seq_labeled_negative.csv"             # 原始CSV文件
# PEP_COL = "peptide"                    # 肽序列列名
# HLA_SEQ_COL = "HLA"           # HLA序列列名
# OUT_CSV = "unique_peptide_hla_negative.csv"     # 输出文件

# # 读取文件
# df = pd.read_csv(CSV_FILE, dtype=str)

# # 去掉空格并统一大小写（可选，防止同义重复）
# df[PEP_COL] = df[PEP_COL].str.strip().str.upper()
# df[HLA_SEQ_COL] = df[HLA_SEQ_COL].str.strip()

# # 按 peptide + HLA_Sequence 去重
# df_unique = df.drop_duplicates(subset=[PEP_COL, HLA_SEQ_COL])

# # 保存结果
# df_unique.to_csv(OUT_CSV, index=False, encoding="utf-8")

# print(f"处理完成：原始 {len(df)} 行，去重后 {len(df_unique)} 行，结果已保存到 {OUT_CSV}")
# import pandas as pd
# from sklearn.model_selection import train_test_split

# # ====== 配置区（把下面两个路径替换成你的文件路径） ======
# pos_path = "/data0/chenkai/data/microLM-main/downstream/Code_antigen_MHC/UnifyImmun-main/data/data_HLA_I_micro/peptide_hla_positive.csv"  # Positive 文件路径
# neg_path = "/data0/chenkai/data/microLM-main/downstream/Code_antigen_MHC/UnifyImmun-main/data/data_HLA_I_micro/peptide_hla_negative.csv"  # Negative 文件路径
# out_all   = "dataset_all.csv"
# out_train = "train.csv"
# out_test  = "test.csv"
# random_state = 42  # 可改，保证可复现
# # ======================================================

# # 读取
# df_pos = pd.read_csv(pos_path)
# df_neg = pd.read_csv(neg_path)

# # 可选：校验列是否一致（除标签外）
# if set(df_pos.columns) != set(df_neg.columns):
#     # 如果两边列顺序不同但集合相同，统一列顺序
#     common_cols = sorted(list(set(df_pos.columns).intersection(set(df_neg.columns))))
#     # 如果你明确两边是同架构，这里也可以直接用 df_pos.columns 作为标准
#     df_pos = df_pos[common_cols]
#     df_neg = df_neg[common_cols]

# # 加标签
# df_pos = df_pos.copy()
# df_neg = df_neg.copy()
# df_pos["label"] = 1
# df_neg["label"] = 0

# # 先做均衡（下采样多数类）
# n_pos = len(df_pos)
# n_neg = len(df_neg)
# n_min = min(n_pos, n_neg)

# df_pos_bal = df_pos.sample(n=n_min, random_state=random_state, replace=False)
# df_neg_bal = df_neg.sample(n=n_min, random_state=random_state, replace=False)

# # 合并成整体均衡数据集
# df_all_bal = pd.concat([df_pos_bal, df_neg_bal], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

# # 拆分 80/20（保证两边正负均衡 -> stratify）
# X = df_all_bal.drop(columns=["label"])
# y = df_all_bal["label"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=random_state
# )

# train_df = X_train.copy()
# train_df["label"] = y_train.values

# test_df = X_test.copy()
# test_df["label"] = y_test.values

# # 保存：整体（均衡后的）、训练集、测试集
# df_all_bal.to_csv(out_all, index=False)
# train_df.to_csv(out_train, index=False)
# test_df.to_csv(out_test, index=False)

# # 打印一下分布确认
# print("Overall (balanced) size:", len(df_all_bal), "pos:", (df_all_bal['label']==1).sum(), "neg:", (df_all_bal['label']==0).sum())
# print("Train size:", len(train_df), "pos:", (train_df['label']==1).sum(), "neg:", (train_df['label']==0).sum())
# print("Test  size:", len(test_df), "pos:", (test_df['label']==1).sum(), "neg:", (test_df['label']==0).sum())



import pandas as pd

# 允许的氨基酸字母（仅大写）
ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWY")

def is_valid_seq(seq: str) -> bool:
    """严格判断氨基酸序列：必须全为大写且仅包含标准氨基酸"""
    if not isinstance(seq, str) or len(seq) == 0:
        return False
    # 不转大写，直接检查，确保大小写严格区分
    return all(ch in ALLOWED_AA for ch in seq)

def clean_dataset(input_path="dataset.csv", 
                  output_clean="dataset_cleaned.csv", 
                  output_removed="dataset_removed.csv"):
    # 读取数据
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)

    # 标准化列名（假设三列分别是 TCR, peptide, label）
    df = df.rename(columns={df.columns[0]: "TCR",
                            df.columns[2]: "peptide",
                            df.columns[3]: "label"})

    # 去掉首尾空格（不改变大小写）
    df["TCR"] = df["TCR"].str.strip()
    df["peptide"] = df["peptide"].str.strip()

    total = len(df)

    # ---- 删除含有小写/非法字符 ----
    mask_valid = df["TCR"].apply(is_valid_seq) & df["peptide"].apply(is_valid_seq)
    removed_invalid = df[~mask_valid]  # 保存被删除的非法行
    df = df[mask_valid]

    # ---- 删除长度 > 34 ----
    mask_len = (df["TCR"].str.len() <= 34) & (df["peptide"].str.len() <= 34)
    removed_too_long = df[~mask_len]  # 保存被删除的过长行
    df = df[mask_len]

    # ---- 合并删除的行 ----
    removed_all = pd.concat([removed_invalid, removed_too_long], axis=0).reset_index(drop=True)

    # ---- 保存结果 ----
    df.reset_index(drop=True).to_csv(output_clean, index=False)
    removed_all.to_csv(output_removed, index=False)

    print("=== 清洗完成 ===")
    print(f"输入总行数: {total}")
    print(f"删除非法字符/含小写行数: {len(removed_invalid)}")
    print(f"删除长度>34行数: {len(removed_too_long)}")
    print(f"输出保留行数: {len(df)}")
    print(f"清洗后数据保存到: {output_clean}")
    print(f"删除的序列保存到: {output_removed}")

if __name__ == "__main__":
    clean_dataset("/data0/chenkai/data/microLM-main/downstream/Code_antigen_MHC/UnifyImmun-main/data/data_HLA_I_micro/micro_set.csv", "dataset_cleaned.csv", "dataset_removed.csv")

# import pandas as pd
# from sklearn.model_selection import StratifiedKFold
# from collections import Counter
# import os

# # 输入输出设置
# INPUT_PATH = "train.csv"       # 你的原始CSV
# OUTPUT_DIR = "cv_splits"         # 输出目录
# N_SPLITS = 5                     # 折数
# RANDOM_STATE = 42                # 随机种子

# # 确保输出目录存在
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # 读取数据
# df = pd.read_csv(INPUT_PATH)

# # 确认包含四列
# required_cols = ["peptide", "MHC_Restriction_Name", "HLA", "label"]
# for col in required_cols:
#     if col not in df.columns:
#         raise ValueError(f"缺少必要列 {col}，当前列为 {list(df.columns)}")

# # label 转换为 int
# df["label"] = df["label"].astype(int)
# if not set(df["label"].unique()).issubset({0, 1}):
#     raise ValueError("label 列必须只包含 0 和 1")

# X = df.drop(columns=["label"])
# y = df["label"]

# # 设置分层五折
# skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# summary = []

# for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
#     train_df = df.iloc[train_idx].reset_index(drop=True)
#     test_df = df.iloc[test_idx].reset_index(drop=True)

#     # 保存每折数据
#     train_path = os.path.join(OUTPUT_DIR, f"train_fold_{fold}.csv")
#     test_path = os.path.join(OUTPUT_DIR, f"test_fold_{fold}.csv")
#     train_df.to_csv(train_path, index=False)
#     test_df.to_csv(test_path, index=False)

#     # 打印分布情况
#     tr_counts = Counter(train_df["label"])
#     te_counts = Counter(test_df["label"])
#     print(f"Fold {fold} - Train: {tr_counts}, Test: {te_counts}")

#     summary.append({
#         "fold": fold,
#         "train_size": len(train_df),
#         "train_label0": tr_counts.get(0, 0),
#         "train_label1": tr_counts.get(1, 0),
#         "test_size": len(test_df),
#         "test_label0": te_counts.get(0, 0),
#         "test_label1": te_counts.get(1, 0),
#     })

# # 保存整体统计
# pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_DIR, "fold_summary.csv"), index=False)
# print(f"\n所有折文件和 fold_summary.csv 已保存到: {OUTPUT_DIR}")

