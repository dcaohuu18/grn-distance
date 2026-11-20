import argparse
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def precision_k(k, pred, gt, largest=False):
  """
    Computes Top-k precision (adapted from gnn_infer.py).
    - largest=False: ranks ascending (smallest distances / closest to control).
    - largest=True:  ranks descending (largest distances / strongest perturbations).
    """
  if k >= len(pred):
    return float("nan")
  if largest:
    pred = -np.asarray(pred)
    gt = -np.asarray(gt)

  tmp = list(zip(gt, pred))
  tmp.sort()
  beta = [(p[1], p[0], i) for i, p in enumerate(tmp)]
  beta.sort()
  hits = sum(1 for i in range(k) if beta[i][2] < k)
  return hits / k


def evaluate_pair(pred, target, label):
  sp = spearmanr(pred, target)
  kt = kendalltau(pred, target)

  print(f"\n=== {label} (n={len(pred)}) ===")
  print(f"Spearman Rank:                {sp.statistic:.4f} (p={sp.pvalue:.2e})")
  print(f"Kendall Tau:                  {kt.statistic:.4f} (p={kt.pvalue:.2e})")
  print(
      f"Top-10 Precision (closest):   {precision_k(10, pred, target, largest=False):.4f}"
  )
  print(
      f"Top-20 Precision (closest):   {precision_k(20, pred, target, largest=False):.4f}"
  )
  print(
      f"Top-10 Precision (divergent): {precision_k(10, pred, target, largest=True):.4f}"
  )
  print(
      f"Top-20 Precision (divergent): {precision_k(20, pred, target, largest=True):.4f}"
  )


def main():
  parser = argparse.ArgumentParser(
      description="Compute ranking metrics between two saved distance CSV files."
  )
  parser.add_argument(
      "--file1",
      type=str,
      default="output/DC_3hr_mapper_random_greed_ged.csv",
      help="CSV containing predicted distances (e.g., Mapper_Inferred_GED).",
  )
  parser.add_argument(
      "--file2",
      type=str,
      default="output/DC_3hr_grn_undirected_distances.csv",
      help="CSV containing reference distances (e.g., WL_Distance).",
  )
  parser.add_argument(
      "--col1",
      type=str,
      default="Mapper_Inferred_GED",
      help="Distance column name in file1.",
  )
  parser.add_argument(
      "--col2",
      type=str,
      default=None,
      help="Specific column in file2 (omit to evaluate all detected distance columns).",
  )
  args = parser.parse_args()

  df1 = pd.read_csv(args.file1)
  df2 = pd.read_csv(args.file2)

  merged = pd.merge(df1, df2, on="Sample_Name", suffixes=("_1", "_2"))
  if merged.empty:
    raise ValueError(
        "No matching 'Sample_Name' entries found between the two files."
    )

  col1 = args.col1 if args.col1 in merged.columns else f"{args.col1}_1"
  preds = merged[col1].to_numpy()

  if (
      "Exact_Aligned_Edge_Edit_Dist" in merged.columns
      and col1 != "Exact_Aligned_Edge_Edit_Dist"
  ):
    evaluate_pair(
        preds,
        merged["Exact_Aligned_Edge_Edit_Dist"].to_numpy(),
        f"{col1} vs. Exact_Aligned_Edge_Edit_Dist",
    )

  if args.col2:
    target_cols = [
        args.col2 if args.col2 in merged.columns else f"{args.col2}_2"
    ]
  else:
    candidate_cols = [
        "WL_Distance",
        "Undirected_Edge_Edit_Distance",
        "Edge_Edit_Distance",
        "GED_Estimate",
    ]
    target_cols = [c for c in candidate_cols if c in merged.columns]

  for col in target_cols:
    evaluate_pair(preds, merged[col].to_numpy(), f"{col1} vs. {col}")


if __name__ == "__main__":
  main()