import csv
import glob
import math
import os
from grakel import Graph
from grakel.kernels import VertexHistogram, WeisfeilerLehman
import pandas as pd

GRN_DIR = "output/DC_3hr_grn"
CONTROL_ADJ_PATH = os.path.join(GRN_DIR, "control_cells_adj.csv")
RESULTS_FILE = "output/DC_3hr_grn_undirected_distances.csv"

# Optional: GRNBoost2 outputs every non-zero edge (often hundreds of thousands).
# Set to e.g. 10000 to keep only top high-confidence edges, or None to use all edges.
TOP_K_EDGES = 2500
WL_ITERATIONS = 3  # Number of neighborhood hops (2-4 is standard)


def load_undirected_grakel_and_edges(csv_path, top_k=None):
  """
    Loads a GRNBoost2 *_adj.csv file as an undirected GraKeL Graph
    (using gene symbols as discrete node labels) and a set of canonical
    undirected edges.
    """
  df = pd.read_csv(csv_path)
  if top_k is not None and len(df) > top_k:
    df = df.nlargest(top_k, "importance")

  tfs = df["TF"].astype(str)
  targets = df["target"].astype(str)
  importances = df["importance"].astype(float)

  nodes = set(tfs).union(set(targets))
  adj_dict = {node: {} for node in nodes}
  edge_set = set()

  for u, v, imp in zip(tfs, targets, importances):
    if u == v:
      continue  # Exclude self-loops for a simple undirected graph

    # Canonical undirected edge representation
    edge = (u, v) if u < v else (v, u)
    edge_set.add(edge)

    # GraKeL requires symmetric entries (u -> v and v -> u) for undirected graphs.
    # If two TFs regulate each other in both directions, keep the stronger weight.
    weight = max(adj_dict[u].get(v, 0.0), imp)
    adj_dict[u][v] = weight
    adj_dict[v][u] = weight

  node_labels = {node: node for node in nodes}
  g_grakel = Graph(adj_dict, node_labels=node_labels)
  return g_grakel, edge_set


def main():
  if not os.path.exists(CONTROL_ADJ_PATH):
    print(f"Control graph not found at {CONTROL_ADJ_PATH}")
    return

  print("Loading undirected control graph...")
  G_ctrl, edges_ctrl = load_undirected_grakel_and_edges(
      CONTROL_ADJ_PATH, top_k=TOP_K_EDGES
  )

  # Initialize and fit the WL Subtree kernel on the control graph once
  wl_kernel = WeisfeilerLehman(
      n_iter=WL_ITERATIONS,
      base_graph_kernel=VertexHistogram,
      normalize=True,
  )
  wl_kernel.fit_transform([G_ctrl])

  adj_files = sorted(glob.glob(os.path.join(GRN_DIR, "*_adj.csv")))

  # Track already-processed samples so interrupted runs can resume cleanly
  completed_samples = set()
  if os.path.isfile(RESULTS_FILE):
    existing_df = pd.read_csv(RESULTS_FILE)
    completed_samples = set(existing_df["Sample_Name"].astype(str))

  with open(RESULTS_FILE, mode="a", newline="") as f:
    writer = csv.writer(f)
    if not completed_samples:
      writer.writerow([
          "Sample_Name",
          "WL_Similarity",
          "WL_Distance",
          "Undirected_Edge_Edit_Distance",
          "Shared_Undirected_Edges",
      ])

    for adj_path in adj_files:
      if os.path.abspath(adj_path) == os.path.abspath(CONTROL_ADJ_PATH):
        continue

      sample_name = os.path.basename(adj_path).replace("_adj.csv", "")
      if sample_name in completed_samples:
        continue

      G_pert, edges_pert = load_undirected_grakel_and_edges(
          adj_path, top_k=TOP_K_EDGES
      )

      # 1. Undirected multi-hop labeled structural similarity & metric distance
      wl_sim = float(wl_kernel.transform([G_pert])[0, 0])
      # Convert normalized WL kernel similarity (wl_sim in [0, 1] with VertexHistogram) to Euclidean distance
      # in the kernel's Reproducing Kernel Hilbert Space (RKHS):
      #
      # Derivation:
      #   With normalize=True, feature vectors phi_1 and phi_2 are L2-normalized to unit length:
      #       ||phi_1||^2 = 1,  ||phi_2||^2 = 1,  and  <phi_1, phi_2> = wl_sim
      #
      #   Expanding the squared Euclidean distance between unit vectors:
      #       d(G1, G2)^2 = ||phi_1 - phi_2||^2
      #                   = ||phi_1||^2 + ||phi_2||^2 - 2 * <phi_1, phi_2>
      #                   = 1 + 1 - 2 * wl_sim
      #                   = 2 * (1 - wl_sim)
      #   Taking the square root yields a proper distance metric in [0, sqrt(2)].
      #   max(0.0, ...) guards against math domain errors when floating-point rounding gives wl_sim > 1.0
      wl_dist = math.sqrt(max(0.0, 2.0 * (1.0 - wl_sim))) 

      # 2. Exact node-aligned undirected edge edit distance
      shared_edges = len(edges_ctrl.intersection(edges_pert))
      edge_edit_dist = len(edges_ctrl.symmetric_difference(edges_pert))

      print(
          f"{sample_name}: WL_Dist={wl_dist:.4f} (Sim={wl_sim:.4f}) | "
          f"Undirected_Edge_Edit_Dist={edge_edit_dist}"
      )

      writer.writerow([
          sample_name,
          round(wl_sim, 6),
          round(wl_dist, 6),
          edge_edit_dist,
          shared_edges,
      ])
      f.flush()

  print("\n--- All undirected GRN pairs processed successfully ---")


if __name__ == "__main__":
  main()