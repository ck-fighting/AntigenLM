# protective_antigen3 real-species dataset README

本文档记录 `Data/protective_antigen3/positive.fasta` 的重新构建流程和本次已经生成的结果。

最终产物为 3 组数据：

```text
Data/protective_antigen3/20_similarity/
Data/protective_antigen3/30_similarity/
Data/protective_antigen3/40_similarity/
```

每个目录内均包含 5 次重复的 cluster-aware `70/15/15` 划分：

```text
train_fold_1.csv  val_fold_1.csv  test_fold_1.csv
...
train_fold_5.csv  val_fold_5.csv  test_fold_5.csv
```

同样的 fold CSV 和统计文件已经同步到训练代码默认读取目录：

```text
Downstream_MR/protective_antigen/data2/20_similarity/
Downstream_MR/protective_antigen/data2/30_similarity/
Downstream_MR/protective_antigen/data2/40_similarity/
```

## 本次新增清洗规则

本次不是简单重复旧版划分，而是在原流程中重新定义了正样本去冗余和清洗规则：

1. 正样本 50% 去冗余改为 MMseqs2 聚类，不再用 CD-HIT 默认最长序列作为代表。
2. 每个 MMseqs2 50% cluster 内对所有成员打代表序列分数，优先选择 header 更像保护性抗原的序列，例如 antigen、immunogen、outer membrane、surface、adhesin、fimbria/pilus、toxin/toxoid、secreted/exported、virulence 等。
3. 代表序列选择时会惩罚明显不像保护性抗原的成员，例如短片段、hypothetical/uncharacterized、DNA 复制/结合、核糖体、tRNA/翻译、中心代谢、转录/修复等。
4. 正样本 NR50 后，继续使用固定人工正样本删表 `positive_manual_exclusions.tsv` 和已有预测观察表 `curation_filters/low_score_positive_observations.tsv` 删除可疑正样本。
5. 本次使用 `--use-existing-curation-reports`，只读取已经固定下来的观察表；没有实时重新读取最新测试预测结果，因此不会把本次新测试结果反向用于构建。
6. 负样本候选池优先保留与抗原无关的负样本，默认删除 header 中含抗原、毒力、表面暴露、膜蛋白、分泌蛋白、fragment、DUF、hypothetical/uncharacterized、管家基因等关键词的负样本。

划分阶段使用 cluster-aware stratification：

1. cluster 分配时继续平衡每个 fold 的正样本数量和真实物种分布。
2. 同一 40% protein cluster 不允许跨 train/val/test。
3. 完全相同的蛋白序列会被强制并入同一个 cluster，避免 exact sequence 跨集合泄漏。
4. 每个 similarity 先固定一份最终数据集：`300` 条正样本 + `3000` 条负样本。
5. 5 个 fold 只从同一批固定 record_id 中按 cluster/bin 取 train、val、test，不再每折重新抽负样本。
6. 上述均衡不放宽每个 split 内正负 `1:10`。

相关报告：

```text
positive_manual_exclusions.tsv
positive_nr50/positive_nr50_representatives.tsv
positive_nr50/positive_nr50_cluster_members.tsv
negative_candidates/reports/negative_curation_removed.tsv
negative_candidates/reports/negative_nr50_summary.tsv
```

历史预测观察文件保留在 `curation_filters/` 中；本次运行显式使用 `--use-existing-curation-reports` 读取这些固定观察表，但未启用 `--use-prediction-curation` 实时读取最新预测文件。

抗原/非理想负样本关键词包括但不限于：

```text
adhesin, antigen, autotransporter, binding protein, capsule, cagA,
duf, exotoxin, exoenzyme, exported, extracellular, fimbria,
fimbrial, fimbrillin, flagellin, flagellar, fragment, hemolysin,
hemagglutinin, hypothetical, ig-like, immunoglobulin, immunogen,
inner membrane, invasion, invasin, lectin, lipoprotein, membrane,
momp, ompA/ompB/ompC, outer membrane, pe-pgrs, pilin, pilus,
porin, ppe, protease, repeat protein, secreted, secretion,
secretory, siderophore receptor, surface, toxin, toxoid,
transporter, uncharacterized, vaccine, virulence
```

注意：本次划分不使用最新测试预测结果做在线筛样本，因此不会因为当前模型结果反向进入数据构建而产生循环评估。低分正样本和高分 FP 负样本只来自固定观察表，后续复现时观察表内容不变。

## 实际运行概况

输入正样本：

```text
Data/protective_antigen3/positive.fasta
```

本次统计结果：

```text
raw positive count                    657
positive NR50 count                   537
included positives after curation      338
excluded positives                    199
included species before final split    58
final eligible species                 56
final positives per dataset           300
negative NR50 candidates           105286
```

排除的正样本原因：

```text
low_score_non_typical_positive                          139
manual_low_score_first_prediction_le_0.5                 76
hypothetical_or_uncharacterized                          26
short_fragment_lt80                                      26
central_metabolism_housekeeping                          19
manual_short_fragment_lt80                               19
manual_hypothetical_or_uncharacterized                   15
trna_or_translation_housekeeping                          7
synthetic_construct                                       7
manual_dna_binding_or_replication_protein                 5
dna_binding_or_replication_protein                        4
manual_central_metabolism_housekeeping                    4
manual_trna_or_translation_housekeeping                   4
ribosomal_protein                                         4
phage_or_virus                                            3
manual_ribosomal_protein                                  2
transcription_or_repair_housekeeping                      2
manual_transcription_or_biosynthesis_housekeeping         1
```

负样本清洗统计：

```text
raw negative records scanned              820789
removed high-score FP records                878
removed antigen-like keyword records      417889
negative NR50 candidates                  105286
split hard negative NR50 candidates            0
```

负样本池：

```text
20_similarity negative_candidate_pool.fasta  103482
30_similarity negative_candidate_pool.fasta  103664
40_similarity negative_candidate_pool.fasta  103942

20_similarity fixed negative_pool.fasta         3000
30_similarity fixed negative_pool.fasta         3000
40_similarity fixed negative_pool.fasta         3000
```

本次严格清洗后进入最终数据集的物种均满足 2% 负样本 buffer 要求：

额外负样本人工清理：从候选池和固定池中移除了 `Antiphagocytic M`，并在
20_similarity 固定负样本池中用同物种、0% positive identity、双向 MMseqs
cross-pool >=40% 命中为 0 的 `Glucose-6-phosphate isomerase` 替换。

```text
negative_available >= ceil(10 * positive_count * 1.02)
```

## 复现命令

主脚本：

```text
Data/protective_antigen3/build_real_species_pipeline.py
```

从项目根目录运行：

```bash
python Data/protective_antigen3/build_real_species_pipeline.py --threads 8
```

重新强制生成中间文件和结果：

```bash
python Data/protective_antigen3/build_real_species_pipeline.py \
  --threads 8 \
  --force \
  --positive-dedup-method mmseqs \
  --use-existing-curation-reports
```

脚本默认关键参数：

```text
thresholds                   20,30,40
negative:positive            10:1
repeat seeds                 42,43,44,45,46
prediction curation          disabled by default
existing curation reports    loaded in this run
positive dedup method        mmseqs
manual positive exclusions   positive_manual_exclusions.tsv
low positive cutoff          y_score <= 0.1 if --use-prediction-curation
high FP negative cutoff      y_score >= 0.9 if --use-prediction-curation
difficulty-aware split       enabled
split low positive cutoff    y_score <= 0.5
split hard negative cutoff   y_score >= 0.5
negative buffer fraction     0.02
BLAST valid hit              evalue <= 1e-5 and qcovs >= 80
split cluster identity       40%
split ratios                 train/val/test = 70/15/15
```

依赖工具：

```text
cd-hit
makeblastdb
blastp
mmseqs
```

## Step 1. 正样本内部 50% 去冗余

使用 MMseqs2 做正样本 50% 聚类：

```bash
mmseqs easy-cluster \
  Data/protective_antigen3/positive_nr50/positive_raw_for_mmseqs.fasta \
  Data/protective_antigen3/positive_nr50/positive_mmseqs_nr50 \
  Data/protective_antigen3/positive_nr50/tmp_positive_mmseqs_nr50 \
  --min-seq-id 0.5 \
  -c 0.8 \
  --cov-mode 0 \
  --cluster-mode 1 \
  --single-step-clustering 1 \
  -s 5.7 \
  --max-seqs 100000 \
  --threads 8
```

MMseqs2 聚类后，脚本不会直接使用 MMseqs2 自带代表序列，而是对每个 cluster 的所有成员重新打分，选出更像保护性抗原的代表序列。

输出：

```text
positive_nr50/positive_nr50.fasta
positive_nr50/positive_raw_header_map.tsv
positive_nr50/positive_mmseqs_nr50_cluster.tsv
positive_nr50/positive_nr50_representatives.tsv
positive_nr50/positive_nr50_cluster_members.tsv
```

## Step 2. 删除非典型或非真实细菌正样本

正样本 NR50 后继续做 metadata 解析和清洗：

1. 删除 synthetic construct。
2. 删除 phage/virus 来源。
3. 删除固定人工删表 `positive_manual_exclusions.tsv` 中记录的可疑正样本。
4. 人工删表主要来自第一次预测低分正样本，以及 header 显示为 hypothetical/uncharacterized、短片段、DNA 复制/结合、核糖体、tRNA/翻译、中心代谢等非典型保护性抗原条目。

输出：

```text
positive_nr50/positive_metadata.tsv
positive_nr50/species_positive_counts.tsv
positive_nr50/excluded_positive_records.tsv
positive_nr50/positive_by_species/*.fasta
```

## Step 3. 按真实物种解析正样本

原始 FASTA header 第一列是旧的大类标签，不是真实物种。因此脚本按以下优先级解析真实物种：

1. `OS=... OX=...`
2. header 末尾方括号中的 organism，例如 `[Yersinia pestis CO92]`
3. PDB/UniProt accession 的少量手动补全
4. 内置 taxid override 表

strain、serovar、subsp. 等会归一化到 species 或 genus，例如：

```text
Yersinia pestis CO92                 -> Yersinia pestis
Escherichia coli O157:H7 str. EDL933 -> Escherichia coli
Streptococcus pyogenes M1 GAS        -> Streptococcus pyogenes
Brucella                             -> Brucella
Treponema                            -> Treponema
```

## Step 4. 下载或复用真实物种蛋白组

负样本来源按真实物种 taxid 构建。

脚本优先复用本项目已有缓存：

```text
Data/protective_antigen/uniprot_raw_by_taxon/
Data/protective_antigen2/uniprot_raw_by_taxon_nr50/
```

如果某个物种没有缓存，或缓存数量不足以满足 10 倍负样本要求，则从 UniProtKB 下载：

```text
taxonomy_id:<taxid>
```

当某个 taxon 查询超过 50000 条 UniProt 记录时，为了可运行性，脚本使用 reviewed subset：

```text
taxonomy_id:<taxid> AND reviewed:true
```

来源报告：

```text
negative_candidates/reports/negative_source_report.tsv
```

## Step 5. 构建抗原无关负样本候选池

每个物种内部先按 accession 去重，然后删除以下负样本：

1. accession 与任意正样本 accession 相同。
2. 氨基酸序列与任意正样本完全相同。
3. 如果显式开启 `--use-prediction-curation`，删除旧预测中 `y_score >= 0.9` 的高分 FP 负样本；本次默认关闭，因此该项删除数为 0。
4. header 中含抗原/表面暴露/毒力/膜蛋白/分泌蛋白/fragment/DUF/hypothetical/uncharacterized/管家功能等关键词的负样本。

输出：

```text
negative_candidates/positive_removed_by_species/*.fasta
negative_candidates/reports/negative_nr50_summary.tsv
negative_candidates/reports/negative_curation_removed.tsv
```

## Step 6. 负样本 50% 去冗余

每个真实物种内部使用 CD-HIT 做 NR50：

```bash
cd-hit \
  -i negative_candidates/positive_removed_by_species/<species>.fasta \
  -o negative_candidates/negative_nr50_by_species/<species>.fasta \
  -c 0.5 \
  -n 3 \
  -G 1 \
  -aS 0.8 \
  -aL 0.8 \
  -g 1 \
  -d 0 \
  -T 8 \
  -M 0
```

输出：

```text
negative_candidates/negative_nr50_by_species/*.fasta
```

## Step 7. 与正样本相似度过滤

使用清洗后的真实细菌正样本建立 BLASTP database：

```text
similarity_to_positive/positive_real_nr50.fasta
similarity_to_positive/blastdb_positive_nr50/
```

所有负样本 NR50 候选与正样本比对，输出：

```text
similarity_to_positive/blast_results/all_negative_vs_positive.tsv
similarity_to_positive/best_positive_hit.tsv
```

有效 hit 定义：

```text
evalue <= 1e-5
qcovs >= 80
```

每个负样本取有效 hit 中最高的 `pident` 作为 `best_positive_pident`。如果没有有效 hit，则记为 `0`。

三组负样本定义：

```text
20_similarity: best_positive_pident < 20
30_similarity: best_positive_pident < 30
40_similarity: best_positive_pident < 40
```

相似度过滤报告：

```text
similarity_to_positive/reports/similarity_filter_summary.tsv
```

## Step 8. 每个物种保持正负 1:10

对每个 threshold 单独判断物种是否满足：

```text
negative_available >= ceil(10 * positive_count * 1.02)
```

满足的物种进入该 threshold 数据集；不足的物种记录在：

```text
<threshold>_similarity/species_eligibility.tsv
```

每个 threshold 目录中的：

```text
positive.fasta
negative_pool.fasta
```

分别是该组划分使用的正样本和候选负样本池。

## Step 9. 固定 40% cluster partition + 五次划分

每个 threshold 数据集内部，先固定最终正负样本池，然后合并正负全部蛋白，只做一次 40% 全蛋白聚类。聚类结果再分配到固定 micro-bin，生成：

```text
<threshold>_similarity/cluster_partition.tsv
```

`cluster_partition.tsv` 记录：

```text
cluster_id
fold_bin
positive_count
negative_count
positive_species_counts
negative_species_counts
fold_1_split ... fold_5_split
```

后续 `train/val/test` 都只从固定 `cluster_id` 对应的 split role 中取样，不再每个 fold 独立随机重新切 cluster。

MMseqs2 40% 聚类命令：

```bash
mmseqs easy-cluster \
  <threshold>_similarity/_mmseqs_cluster40/all_for_cluster40.fasta \
  <threshold>_similarity/_mmseqs_cluster40/cluster40 \
  <threshold>_similarity/_mmseqs_cluster40/tmp \
  --min-seq-id 0.4 \
  -s 5.7 \
  -c 0.8 \
  --cov-mode 2 \
  --cluster-mode 1 \
  --single-step-clustering 1 \
  --max-seqs 100000 \
  --threads 8
```

划分规则：

1. 以固定的 40% protein cluster/bin 为单位划分。
2. 同一个 cluster 不允许同时出现在 train、val、test。
3. 5 次重复使用同一个 `cluster_partition.tsv` 中的固定 split role。
4. 每个 similarity 先按 species/bin 固定 `3000` 条最终负样本；5 个 fold 共用这同一批负样本 record_id。
5. 固定负样本优先选择低 `best_positive_pident`、低历史预测分数、非 hard-negative、无抗原关键词命中的蛋白，避免把很可能是保护性抗原的蛋白放入负样本。
6. 每个 split 内总正负比例严格为 `1:10`。
7. 按真实物种填充负样本，使 split 内每个出现正样本的物种也严格保持 `1:10`。
8. 正样本 cluster 分配目标包含：总正样本数、真实物种分布。
9. 若显式启用预测清洗并存在历史分数，才额外均衡 low-score positive 和 hard negative；本次固定负样本池未选入 hard negative。
10. 如果任何 difficulty 目标和严格 `1:10` 冲突，优先保证 `1:10` 和 cluster 不跨 split。

随机性说明：

1. `train`、`val`、`test` 是随机划分的，但不是逐条蛋白简单随机打散。
2. 划分前先做 40% 全蛋白相似性聚类，然后将 cluster 固定分配到 micro-bin 和 fold split role，减少同源蛋白跨集合泄漏。
3. 5 次重复使用固定随机种子 `42,43,44,45,46`，且共享固定 `cluster_partition.tsv`，因此划分结果可复现。
4. 负样本池只固定抽取一次；之后 5 个 fold 不再重新抽负样本，只根据固定 cluster/bin split role 改变 train、val、test 归属。
5. 本次没有使用动态预测分数参与划分；固定负样本池中 `hard_negative_count` 为 0。
6. `cluster_overlap_check.tsv` 用于检查同一 cluster 是否跨 train/val/test；本次三组数据均为 0 overlap。
7. `cross_split_similarity_check.tsv` 使用 MMseqs2 对 `train-vs-val`、`train-vs-test`、`val-vs-test` 做额外检查；候选命中需同时满足 `pident >= 40%`、`qcov >= 0.8`、`tcov >= 0.8` 才计为全蛋白泄漏。

输出字段：

```text
sequence
label
record_id
species
taxid
protein_cluster_40
best_positive_pident
source_header
split
repeat
prediction_difficulty_score
difficulty_group
```

其中：

```text
label = 1  positive
label = 0  negative
difficulty_group = low_score_positive / standard_positive / hard_negative / standard_negative
```

## 每个输出目录的文件说明

以 `20_similarity/` 为例，`30_similarity/` 和 `40_similarity/` 相同：

```text
README.txt                  简短说明
positive.fasta              本组使用的正样本
negative_candidate_pool.fasta  相似度和物种过滤后的候选负样本池
negative_pool.fasta         本组最终固定负样本池，3000 条
fixed_negative_pool.tsv     最终固定负样本明细
fixed_negative_pool_summary.tsv  每个 species/bin 的负样本配额和选择统计
cluster40_assignments.tsv   40% cluster 分配表
cluster_partition.tsv       固定 cluster_id -> fold_bin/split_role 分配表
species_eligibility.tsv     物种是否满足 10 倍负样本和 2% buffer
split_summary.tsv           每次 train/val/test 总体统计，包含 low-score positive 和 hard negative 数量
species_split_summary.tsv   每次 split 内分物种统计，包含 low-score positive 和 hard negative 数量
cluster_overlap_check.tsv   cluster 跨 split 泄漏检查
cross_split_similarity_check.tsv  train/val/test 跨 split 全蛋白相似性检查
cross_split_similarity_pairs.tsv  若存在跨 split 全蛋白命中，记录具体 pair
split_warnings.tsv          无法满足 1:10 时的警告
train_fold_1.csv
val_fold_1.csv
test_fold_1.csv
...
train_fold_5.csv
val_fold_5.csv
test_fold_5.csv
```

本次核对结果：

```text
20_similarity  split CSV files 15  fixed negatives 3000  cluster overlaps 0  warnings 0  bad ratios 0
30_similarity  split CSV files 15  fixed negatives 3000  cluster overlaps 0  warnings 0  bad ratios 0
40_similarity  split CSV files 15  fixed negatives 3000  cluster overlaps 0  warnings 0  bad ratios 0
```

每个 fold 的正负比例均为 `1:10`，5 个 fold 共用同一批 `300` 正样本和 `3000` 负样本，且 full-protein cross-split similarity hit 均为 0。fold 1 示例：

```text
20_similarity fold 1:
train  positive 212  negative 2120  low_pos 0  hard_neg 0
val    positive 44   negative 440   low_pos 0  hard_neg 0
test   positive 44   negative 440   low_pos 0  hard_neg 0

30_similarity fold 1:
train  positive 212  negative 2120  low_pos 0  hard_neg 0
val    positive 44   negative 440   low_pos 0  hard_neg 0
test   positive 44   negative 440   low_pos 0  hard_neg 0

40_similarity fold 1:
train  positive 212  negative 2120  low_pos 0  hard_neg 0
val    positive 44   negative 440   low_pos 0  hard_neg 0
test   positive 44   negative 440   low_pos 0  hard_neg 0
```

完整统计见：

```text
20_similarity/split_summary.tsv
30_similarity/split_summary.tsv
40_similarity/split_summary.tsv
```

跨 split 全蛋白相似性检查见：

```text
20_similarity/cross_split_similarity_check.tsv
30_similarity/cross_split_similarity_check.tsv
40_similarity/cross_split_similarity_check.tsv
```

## 验收检查命令

检查每组是否都有 15 个 split CSV：

```bash
for d in 20_similarity 30_similarity 40_similarity; do
  find Data/protective_antigen3/$d -maxdepth 1 -name '*_fold_*.csv' | wc -l
done
```

检查 cluster 是否跨 train/val/test：

```bash
for d in 20_similarity 30_similarity 40_similarity; do
  wc -l Data/protective_antigen3/$d/cluster_overlap_check.tsv
done
```

只有表头表示没有 overlap。

检查划分统计：

```bash
cat Data/protective_antigen3/20_similarity/split_summary.tsv
cat Data/protective_antigen3/30_similarity/split_summary.tsv
cat Data/protective_antigen3/40_similarity/split_summary.tsv
```

检查不足物种：

```bash
for d in 20_similarity 30_similarity 40_similarity; do
  awk -F'\t' 'NR==1 || $6+0==0' Data/protective_antigen3/$d/species_eligibility.tsv
done
```

本次不足物种为：

```text
Chlamydia trachomatis
Streptococcus pyogenes
```

## 最终数据校验

2026-05-29 对 `Data/protective_antigen3` 和实际训练目录
`Downstream_MR/protective_antigen/data2` 做了完整一致性检查：

```text
PASS 493
WARN 0
FAIL 0
```

校验脚本和报告：

```text
Data/protective_antigen3/validate_final_dataset.py
Data/protective_antigen3/final_dataset_validation_report.tsv
Data/protective_antigen3/final_dataset_validation_summary.tsv
Downstream_MR/protective_antigen/data2/final_dataset_validation_report.tsv
Downstream_MR/protective_antigen/data2/final_dataset_validation_summary.tsv
```

核心校验项包括：

```text
1. Data/protective_antigen3 与 Downstream_MR/protective_antigen/data2 的可交付文件 MD5 完全一致
2. 每个 similarity 固定同一批 300 正样本 + 3000 负样本，5 个 fold 不重新抽负样本
3. 每个 fold 内 train/val/test 正负比例均为 1:10
4. 每个 fold 内 protein_cluster_40 不跨 train/val/test
5. cluster_partition.tsv 中的 split_role 与 CSV 中每条记录一致
6. cross_split_similarity_check.tsv 中 directional_hit_count 全部为 0
7. 固定负样本中没有高风险抗原相关关键词命中
8. 正负样本没有 exact sequence overlap
9. 20/30/40 三组使用同一批最终正样本
```

最终负样本与正样本的最高 BLASTP identity 分布：

```text
20_similarity: negatives=3000  max_identity=0.000   buckets={'0-10': 3000}
30_similarity: negatives=3000  max_identity=29.688  buckets={'0-10': 2964, '20-30': 36}
40_similarity: negatives=3000  max_identity=39.884  buckets={'0-10': 2885, '10-20': 1, '20-30': 40, '30-40': 74}
```

## 注意事项

1. 这里的 `20_similarity`、`30_similarity`、`40_similarity` 表示负样本与任意清洗后正样本最高有效 BLASTP identity 分别小于 20%、30%、40%。
2. 物种使用真实 scientific name/taxid，不使用原始 header 第一列的大类标签。
3. 对记录数很大的 UniProt taxon，本次使用 reviewed subset 下载；这是为了保证流程可完成，相关查询记录保存在 `uniprot_raw_by_species/<species>/*.query.txt`。
4. 本次删除低分正样本和抗原样负样本是为了降低标签噪声；固定删表和关键词规则已经写入脚本参数和 TSV 报告，后续可复现。
5. 若必须保留 `Chlamydia trachomatis` 或 `Streptococcus pyogenes`，需要额外增加这些物种的干净负样本来源，或放宽关键词过滤/2% buffer/cluster-aware 约束；当前结果没有自动放宽 `1:10`。
