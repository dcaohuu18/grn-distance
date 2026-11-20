import argparse
import csv
import glob
import os
import random
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
import torch
from torch_geometric.data import Data


def setup_ged_repo(repo_dir):
  repo_dir = os.path.abspath(repo_dir)
  if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)


def load_undirected_edge_df(csv_path, top_k=None):
  """Loads a GRNBoost2 *_adj.csv file and collapses it into unique undirected edges."""
  df = pd.read_csv(csv_path)
  if top_k is not None and len(df) > top_k:
    df = df.nlargest(top_k, "importance")

  tfs = df["TF"].astype(str)
  targets = df["target"].astype(str)

  undirected_edges = set()
  for u, v in zip(tfs, targets):
    if u != v:
      undirected_edges.add((u, v) if u < v else (v, u))
  return undirected_edges


def build_pyg_undirected_graph(undirected_edges, gene_to_idx, use_gene_labels=True):
  """
    Converts undirected edges into a PyG Data object formatted for mapper.py:
      - Symmetric edge_index ([u, v] and [v, u]) for undirected edge counting
      - One-hot node features g.x (if use_gene_labels=True) so np.argmax(g.x, axis=1)
        recovers exact gene IDs, or 1D constant features (dims=[1, 64, 64]) otherwise.
    """
  active_genes = sorted({g for edge in undirected_edges for g in edge})
  local_map = {gene: i for i, gene in enumerate(active_genes)}
  num_nodes = len(active_genes)

  if use_gene_labels:
    num_classes = len(gene_to_idx)
    x = torch.zeros((num_nodes, num_classes), dtype=torch.float32)
    for i, gene in enumerate(active_genes):
      x[i, gene_to_idx[gene]] = 1.0
  else:
    x = torch.ones((num_nodes, 1), dtype=torch.float32)

  if undirected_edges:
    src, dst = [], []
    for u, v in undirected_edges:
      iu, iv = local_map[u], local_map[v]
      src.extend([iu, iv])
      dst.extend([iv, iu])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
  else:
    edge_index = torch.empty((2, 0), dtype=torch.long)

  return Data(x=x, edge_index=edge_index)


@torch.inference_mode()
def get_layerwise_embeddings(pyg_graph, emb_mod, use_layers, device):
  """Extracts node embeddings for layers 1..use_layers from a random GreedEmbedModel."""
  emb_mod.eval()
  g = pyg_graph.to(device)
  layer_embs = []
  for l in range(1, use_layers + 1):
    # Move embeddings back to CPU to keep GPU VRAM minimal
    layer_embs.append(emb_mod(g, l).cpu())
  return layer_embs


def infer_pair_ged_mapper(
    g_ctrl_pyg,
    g_pert_pyg,
    ctrl_layer_embs,
    emb_mod,
    use_layers,
    mat_method,
    nei_delta,
    lab_delta,
    deg_delta,
    device,
    nei_bias_matching,
    hung_matching,
    backward_pass,
    calculate_ged_from_matchings,
    NumpyGraph,
):
  """Applies mapper.py's layerwise node matching + backward refinement to one graph pair."""
  pert_layer_embs = get_layerwise_embeddings(
      g_pert_pyg, emb_mod, use_layers, device
  )
  np_pair = [(NumpyGraph(g_ctrl_pyg), NumpyGraph(g_pert_pyg))]

  layer_preds = []
  layer_dmatrices = []

  for l_idx in range(use_layers):
    # Equivalent to torch.norm(e1.unsqueeze(1) - e2.unsqueeze(0), dim=2) without 3D blowup
    d_mtx = torch.cdist(ctrl_layer_embs[l_idx], pert_layer_embs[l_idx]).numpy()
    layer_dmatrices.append(d_mtx)

    if mat_method == "nei_bias":
      # Pass [d_mtx.copy()] because nei_bias_matching mutates d_mtx in-place
      matchings = nei_bias_matching(
          [d_mtx.copy()],
          np_pair,
          nei_delta,
          lab_delta,
          deg_delta,
          start_pairs=None,
      )
    elif mat_method == "hung":
      matchings = hung_matching([d_mtx])
    else:
      raise ValueError(f"Unsupported mat_method: {mat_method}")

    pred = float(calculate_ged_from_matchings(np_pair, matchings)[0, 0])
    layer_preds.append(pred)

  best_idx = int(np.argmin(layer_preds))
  best_pred = layer_preds[best_idx]

  if mat_method == "nei_bias":
    best_dm = layer_dmatrices[best_idx]
    best_fwd_match = nei_bias_matching(
        [best_dm.copy()],
        np_pair,
        nei_delta,
        lab_delta,
        deg_delta,
        start_pairs=None,
    )
    bwd_pred = float(
        backward_pass(
            [best_dm.copy()],
            np_pair,
            best_fwd_match,
            nei_delta,
            lab_delta,
            deg_delta,
        )[0, 0]
    )
    best_pred = min(best_pred, bwd_pred)

  return best_pred


def main():
  parser = argparse.ArgumentParser(
      description="Infer GRN GED using mapper.py with a randomly initialized GreedEmbedModel."
  )
  parser.add_argument("--ged_repo_dir", type=str, default="ged_distill_new")
  parser.add_argument("--grn_dir", type=str, default="output/DC_3hr_grn")
  parser.add_argument(
      "--results_file",
      type=str,
      default="output/DC_3hr_mapper_random_greed_ged.csv",
  )
  parser.add_argument(
      "--ref_distances_file",
      type=str,
      default="output/DC_3hr_grn_undirected_distances.csv",
      help="Optional CSV from previous step to compare ranking against full-graph metrics.",
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--top_k_edges", type=int, default=2500)
  parser.add_argument("--n_layers", type=int, default=8)
  parser.add_argument("--hidden_dim", type=int, default=64)
  parser.add_argument("--output_dim", type=int, default=64)
  parser.add_argument("--use_layers", type=int, default=8)
  parser.add_argument(
      "--mat_method", type=str, default="nei_bias", choices=["nei_bias", "hung"]
  )
  parser.add_argument("--nei_delta", type=float, default=5.0)
  parser.add_argument("--lab_delta", type=float, default=1.0)
  parser.add_argument("--deg_delta", type=float, default=2.5)
  parser.add_argument(
      "--unlabeled",
      action="store_true",
      help="Use 1D constant features (dims=[1, 64, 64]) instead of gene one-hot labels.",
  )
  cmdl_args = parser.parse_args()

  setup_ged_repo(cmdl_args.ged_repo_dir)
  from gnn_infer import precision_k
  from mapper import (
      GreedEmbedModel,
      NumpyGraph,
      backward_pass,
      calculate_ged_from_matchings,
      hung_matching,
      nei_bias_matching,
  )

  # Set seeds for reproducible random GreedEmbedModel initialization & tie-breaking
  random.seed(cmdl_args.seed)
  np.random.seed(cmdl_args.seed)
  torch.manual_seed(cmdl_args.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(cmdl_args.seed)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  control_path = os.path.join(cmdl_args.grn_dir, "control_cells_adj.csv")
  if not os.path.exists(control_path):
    raise FileNotFoundError(f"Control GRN not found at {control_path}")

  adj_files = sorted(glob.glob(os.path.join(cmdl_args.grn_dir, "*_adj.csv")))
  pert_files = [
      p for p in adj_files if os.path.abspath(p) != os.path.abspath(control_path)
  ]

  print(f"Loading undirected edge sets (top_k_edges={cmdl_args.top_k_edges})...")
  ctrl_edges = load_undirected_edge_df(control_path, top_k=cmdl_args.top_k_edges)
  pert_edges_map = {}
  all_genes = {g for e in ctrl_edges for g in e}

  for p in pert_files:
    sample_name = os.path.basename(p).replace("_adj.csv", "")
    edges = load_undirected_edge_df(p, top_k=cmdl_args.top_k_edges)
    pert_edges_map[sample_name] = edges
    all_genes.update(g for e in edges for g in e)

  gene_to_idx = {g: i for i, g in enumerate(sorted(all_genes))}
  use_gene_labels = not cmdl_args.unlabeled
  input_dim = len(gene_to_idx) if use_gene_labels else 1
  lab_delta = cmdl_args.lab_delta if use_gene_labels else 0.0

  print(
      f"Initializing random GreedEmbedModel(n_layers={cmdl_args.n_layers}, "
      f"dims=[{input_dim}, {cmdl_args.hidden_dim}, {cmdl_args.output_dim}]) on {device}..."
  )
  emb_mod = GreedEmbedModel(
      cmdl_args.n_layers, input_dim, cmdl_args.hidden_dim, cmdl_args.output_dim
  ).to(device)
  emb_mod.eval()

  g_ctrl_pyg = build_pyg_undirected_graph(
      ctrl_edges, gene_to_idx, use_gene_labels=use_gene_labels
  )
  ctrl_layer_embs = get_layerwise_embeddings(
      g_ctrl_pyg, emb_mod, cmdl_args.use_layers, device
  )

  # Resume support if results file already exists
  completed = set()
  if os.path.isfile(cmdl_args.results_file):
    prev_df = pd.read_csv(cmdl_args.results_file)
    completed = set(prev_df["Sample_Name"].astype(str))

  with open(cmdl_args.results_file, mode="a", newline="") as f:
    writer = csv.writer(f)
    if not completed:
      writer.writerow([
          "Sample_Name",
          "Mapper_Inferred_GED",
          "Num_Nodes_Pert",
          "Num_Edges_Pert",
          "Infer_Seconds",
      ])

    for sample_name, pert_edges in pert_edges_map.items():
      if sample_name in completed:
        continue

      t0 = time.perf_counter()
      g_pert_pyg = build_pyg_undirected_graph(
          pert_edges, gene_to_idx, use_gene_labels=use_gene_labels
      )
      inferred_ged = infer_pair_ged_mapper(
          g_ctrl_pyg=g_ctrl_pyg,
          g_pert_pyg=g_pert_pyg,
          ctrl_layer_embs=ctrl_layer_embs,
          emb_mod=emb_mod,
          use_layers=cmdl_args.use_layers,
          mat_method=cmdl_args.mat_method,
          nei_delta=cmdl_args.nei_delta,
          lab_delta=lab_delta,
          deg_delta=cmdl_args.deg_delta,
          device=device,
          nei_bias_matching=nei_bias_matching,
          hung_matching=hung_matching,
          backward_pass=backward_pass,
          calculate_ged_from_matchings=calculate_ged_from_matchings,
          NumpyGraph=NumpyGraph,
      )
      elapsed = time.perf_counter() - t0

      print(
          f"{sample_name}: Mapper_GED={inferred_ged:.1f} "
      )
      writer.writerow([
          sample_name,
          inferred_ged,
          g_pert_pyg.num_nodes,
          len(pert_edges),
          round(elapsed, 4),
      ])
      f.flush()

  # --- Compute Ranking Coefficients (matching mapper.py / gnn_infer.py) ---
  res_df = pd.read_csv(cmdl_args.results_file)
  preds = res_df["Mapper_Inferred_GED"].to_numpy()

  # Compare against full-graph WL_Distance / Undirected_Edge_Edit_Distance if present
  if os.path.isfile(cmdl_args.ref_distances_file):
    ref_df = pd.read_csv(cmdl_args.ref_distances_file)
    merged = pd.merge(res_df, ref_df, on="Sample_Name")
    if len(merged) >= 3:
      m_preds = merged["Mapper_Inferred_GED"].to_numpy()
      for col in ["WL_Distance", "Undirected_Edge_Edit_Distance"]:
        if col in merged.columns:
          target = merged[col].to_numpy()
          print(f"\n=== Ranking Evaluation vs. Full-Graph {col} ===")
          print(f"Spearman Rank:    {spearmanr(m_preds, target).statistic:.4f}")
          print(f"Kendall Tau:      {kendalltau(m_preds, target).statistic:.4f}")


if __name__ == "__main__":
  main()

"""
python3 infer_ged.py --ged_repo_dir ../ged_distill_new --grn_dir output/DC_3hr_grn --seed 0 --top_k_edges 2500 --use_layers 8 --mat_method nei_bias --nei_delta 5.0 --lab_delta 1.0 --deg_delta 2.5
"""