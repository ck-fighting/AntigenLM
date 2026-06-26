# pTCR2 Data

This directory stores the datasets used by the pTCR2 downstream task.

## Naming

The current dataset names are:

- `CMA`: the complete CMA benchmark split.
- `Seen`: the seen-peptide split used for model training and cross-validation.
- `Unseen`: the unseen-peptide evaluation split.

## Files

| File or directory | Description |
| --- | --- |
| `CMA_5fold_splits/` | Five-fold split for the CMA setting. |
| `Seen.csv` | Full Seen dataset. |
| `Seen_5fold_splits/` | Five-fold split used by `TCR_train.py` and `TCR_test.py` by default. |
| `Seen_middle_Unseen.csv` | Combined Seen, middle-frequency, and Unseen dataset retained for analysis. |
| `Unseen.csv` | Unseen evaluation set. |
| `Covid_set.csv` | External COVID evaluation set tested with CMA checkpoints. |
| `middle.csv` | Middle-frequency peptide subset retained for analysis. |
| `data_dict.npy` | Auxiliary dataset dictionary used by legacy utilities. |

## Default Script Paths

`../TCR_train.py` now defaults to:

```text
data/Seen_5fold_splits/
../trained_model/pTCR3/Seen/
```

`../TCR_test.py` now defaults to:

```text
../trained_model/pTCR3/Seen/
../result/pTCR3/AntigenLM_Seen/
```

When `--cv_dir` is omitted, `../TCR_test.py` infers `data/Seen_5fold_splits/` for Seen weights and `data/CMA_5fold_splits/` for CMA weights.
The scripts select `data_cached_seen_5fold/` or `data_cached_cma_5fold/` automatically from the selected fold split.
External datasets are only evaluated when `--independent_csv` or `--unseen_csv` is provided.
