# import argparse, os, random
# import numpy as np
# import pandas as pd

# def set_seed(seed=22):
#     random.seed(seed); np.random.seed(seed)

# def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
#     col_map = {}
#     if 'Peptide' in df.columns: col_map['Peptide'] = 'antigen'
#     if 'CDR3'    in df.columns: col_map['CDR3']    = 'TCR'
#     if 'Label'   in df.columns: col_map['Label']   = 'label'
#     df = df.rename(columns=col_map)
#     need = {'antigen','TCR','label'}
#     if not need.issubset(df.columns):
#         raise ValueError(f"列名不齐：需要 {need}，实际 {set(df.columns)}，支持的原列名有 Peptide/CDR3/Label 或 antigen/TCR/label")
#     return df[['antigen','TCR','label']].copy()

# def is_labeled(x):
#     return (pd.notna(x)) and (str(x).lower()!='unknown')

# def split_dataset(
#     csv_path: str,
#     out_dir: str = './splits',
#     seed: int = 22,
#     K_shot: int = 2,
#     N_train_min: int = 5,      # 训练阈值：≥5 进训练，否则进 zero 池
#     N_majority: int = 100,
#     majority_test_ratio: float = 0.2
# ):
#     os.makedirs(out_dir, exist_ok=True)
#     set_seed(seed)
#     rng = np.random.default_rng(seed)

#     df_raw = pd.read_csv(csv_path)
#     df = normalize_columns(df_raw)
#     df['is_labeled'] = df['label'].apply(is_labeled)

#     # 每抗原“已标注”样本数
#     labeled = df[df['is_labeled']]
#     cnt_labeled = labeled.groupby('antigen', as_index=False).size().rename(columns={'size':'n_labeled'})

#     # 标注属性
#     meta = cnt_labeled.copy()
#     meta['is_majority']  = meta['n_labeled'] >= N_majority
#     meta['is_trainable'] = (meta['n_labeled'] >= N_train_min) & (~meta['is_majority'])

#     majority_antigens  = set(meta.loc[meta['is_majority'], 'antigen'])
#     trainable_antigens = set(meta.loc[meta['is_trainable'], 'antigen'])

#     # ---------------- Majority 场景 ----------------
#     df_majority = df[df['antigen'].isin(majority_antigens)].copy()
#     maj_train_parts, maj_test_parts = [], []
#     for ag, g in df_majority.groupby('antigen'):
#         idx = np.arange(len(g)); rng.shuffle(idx)
#         cut = int(round(len(g) * (1 - majority_test_ratio)))
#         maj_train_parts.append(g.iloc[idx[:cut]])
#         maj_test_parts.append(g.iloc[idx[cut:]])
#     df_maj_train = pd.concat(maj_train_parts, ignore_index=True) if maj_train_parts else pd.DataFrame(columns=df.columns)
#     df_maj_test  = pd.concat(maj_test_parts,  ignore_index=True) if maj_test_parts  else pd.DataFrame(columns=df.columns)
#     df_maj_train[['antigen','TCR','label']].to_csv(os.path.join(out_dir,'majority_train.csv'), index=False)
#     df_maj_test [['antigen','TCR','label']].to_csv(os.path.join(out_dir,'majority_test.csv'),  index=False)

#     # ---------------- Backbone 训练集（非 majority，且已标注数 ≥ N_train_min） ----------------
#     df_train_backbone = df[df['antigen'].isin(trainable_antigens)].copy()
#     df_train_backbone[['antigen','TCR','label']].to_csv(os.path.join(out_dir,'train_backbone.csv'), index=False)

#     # ---------------- Zero-shot 池（已标注数 < N_train_min） ----------------
#     zero_antigens = set(cnt_labeled.loc[cnt_labeled['n_labeled'] < N_train_min, 'antigen'])
#     zero_pool = df[df['antigen'].isin(zero_antigens)].copy()
#     zero_pool[['antigen','TCR','label']].to_csv(os.path.join(out_dir,'zero_shot.csv'), index=False)  # 按你的口径：全量 zero

#     # ---------------- 从 zero-shot 派生 few-shot ----------------
#     fs_sup_parts, fs_qry_parts = [], []
#     for ag, g in zero_pool.groupby('antigen'):
#         g_labeled   = g[g['is_labeled']]
#         g_unlabeled = g[~g['is_labeled']]

#         n_lab = len(g_labeled)
#         if n_lab > 0:
#             take = min(K_shot, n_lab)          # 不足K就尽量取
#             idx = np.arange(n_lab); rng.shuffle(idx)
#             sup_idx = idx[:take]
#             qry_labeled_idx = idx[take:]
#             fs_sup_parts.append(g_labeled.iloc[sup_idx])
#             # 该抗原剩余（剩余已标注 + 全部 Unknown）为 query
#             fs_qry_parts.append(pd.concat([g_labeled.iloc[qry_labeled_idx], g_unlabeled], ignore_index=True))
#         else:
#             # 完全无标注：support 为空，全部进 query
#             fs_qry_parts.append(g)

#     df_fs_sup = pd.concat(fs_sup_parts, ignore_index=True) if fs_sup_parts else pd.DataFrame(columns=df.columns)
#     df_fs_qry = pd.concat(fs_qry_parts, ignore_index=True) if fs_qry_parts else pd.DataFrame(columns=df.columns)

#     df_fs_sup[['antigen','TCR','label']].to_csv(os.path.join(out_dir,'few_shot_support.csv'), index=False)
#     df_fs_qry[['antigen','TCR','label']].to_csv(os.path.join(out_dir,'few_shot_query.csv'),   index=False)

#     # ---------------- 统计 ----------------
#     def cnt(df_): return len(df_) if df_ is not None else 0
#     few_ags = set(pd.concat([df_fs_sup, df_fs_qry], ignore_index=True)['antigen'].unique()) if cnt(df_fs_sup)+cnt(df_fs_qry)>0 else set()

#     print(f"[Summary @ {out_dir}]")
#     print(f"  Majority antigens: {len(majority_antigens)} | train rows: {cnt(df_maj_train)} | test rows: {cnt(df_maj_test)}")
#     print(f"  Backbone-train antigens (n_labeled≥{N_train_min}, non-majority): {len(trainable_antigens)} | rows: {cnt(df_train_backbone)}")
#     print(f"  Zero-shot antigens (n_labeled<{N_train_min}): {zero_pool['antigen'].nunique()} | rows: {cnt(zero_pool)}")
#     print(f"  Few-shot (derived from zero-shot): support rows: {cnt(df_fs_sup)} | query rows: {cnt(df_fs_qry)} | antigens: {len(few_ags)}")
#     # 提示重叠（设计如此）
#     inter = few_ags & set(zero_pool['antigen'].unique())
#     print(f"  注意：few 与 zero 抗原集合重叠（设计如此）。重叠抗原数: {len(inter)}")

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--csv", required=True, help="path to dataset_all.csv")
#     ap.add_argument("--out_dir", default="./splits", help="output directory")
#     ap.add_argument("--seed", type=int, default=22)
#     ap.add_argument("--k_shot", type=int, default=2)
#     ap.add_argument("--n_train_min", type=int, default=5)
#     ap.add_argument("--n_majority", type=int, default=100)
#     ap.add_argument("--majority_test_ratio", type=float, default=0.2)
#     args = ap.parse_args()
#     split_dataset(
#         csv_path=args.csv,
#         out_dir=args.out_dir,
#         seed=args.seed,
#         K_shot=args.k_shot,
#         N_train_min=args.n_train_min,
#         N_majority=args.n_majority,
#         majority_test_ratio=args.majority_test_ratio
#     )


import pandas as pd

csv_path = "dataset_all.csv"  # 改成你的路径
df = pd.read_csv(csv_path)

# 兼容列名
label_col = "label" if "label" in df.columns else ("Label" if "Label" in df.columns else None)
if label_col is None:
    raise ValueError("找不到标签列：需要 'label' 或 'Label'")

# 只保留真值 0/1，排除 Unknown/NaN/其他
lab = df[label_col].astype(str).str.strip().str.lower()
mask01 = lab.isin(["0","1"])
lab01 = lab[mask01].astype(int)

n_pos = (lab01 == 1).sum()
n_neg = (lab01 == 0).sum()
n_all = n_pos + n_neg
pos_rate = n_pos / n_all if n_all > 0 else 0

print(f"label=1 (正例) 数量: {n_pos}")
print(f"label=0 (负例) 数量: {n_neg}")
print(f"已标注样本总数: {n_all}")
print(f"正例占比: {pos_rate:.3f}")

# 如需顺手看一眼是否夹杂了 Unknown：
n_unknown = (~mask01).sum()
print(f"非0/1标签（如 Unknown）数量: {n_unknown}")
