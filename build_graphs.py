import os
import glob
import subprocess

# Paths to reference databases
TF_LIST = "scenic_ref/mm_tfs.txt"
MOTIF_ANNOT = "scenic_ref/motifs-v9-nr.mgi-m0.001-o0.0.tbl"
FEATHER_DB = "scenic_ref/mm10__refseq-r80__10kb_up_and_down_tss.mc9nr.genes_vs_motifs.rankings.feather"

# Directories
LOOM_DIR = "output/DC_3hr_pre"
GRN_DIR = "output/DC_3hr_grn"
os.makedirs(GRN_DIR, exist_ok=True)

# Find all loom files generated in Step 1
loom_files = glob.glob(os.path.join(LOOM_DIR, "*.loom"))
print(f"Found {len(loom_files)} loom files to process.")

NUM_WORKERS = 4  # Adjust based on available CPU cores

for loom_path in loom_files:
    sample_name = os.path.basename(loom_path).replace(".loom", "")
    
    adj_out = os.path.join(GRN_DIR, f"{sample_name}_adj.csv")
    ctx_out = os.path.join(GRN_DIR, f"{sample_name}_regulons.csv")
    
    # Skip if graph already generated
    if os.path.exists(ctx_out):
        print(f"[{sample_name}] Graph already exists, skipping.")
        continue
        
    print(f"\n==========================================")
    print(f" Processing: {sample_name}")
    print(f"==========================================")
    
    # Step 2a: Run GRNBoost2 (Co-expression)
    cmd_grn = [
        "pyscenic", "grn",
        loom_path,
        TF_LIST,
        "-o", adj_out,
        "--num_workers", str(NUM_WORKERS)
    ]
    print(f"Running GRNBoost2 for {sample_name}...")
    subprocess.run(cmd_grn, check=True)
    
    # Step 2b: Run cisTarget (Motif Pruning to get Directed Edges)
    cmd_ctx = [
        "pyscenic", "ctx",
        adj_out,
        FEATHER_DB,
        "--annotations_fname", MOTIF_ANNOT,
        "--expression_mtx_fname", loom_path,
        "--mode", "custom_multiprocessing",
        "-o", ctx_out,
        "--num_workers", str(NUM_WORKERS)
    ]
    print(f"Running cisTarget for {sample_name}...")
    subprocess.run(cmd_ctx, check=True)

print(f"\nStep 2 Complete! All GRN graph CSVs generated in '{GRN_DIR}'.")