import os
import csv
import gzip
import pandas as pd
import scanpy as sc
from scipy.io import mmread

# Arguments:
cell_fn = "data/DC_3hr/GSM2396856_dc_3hr_cellnames.csv.gz"
gene_fn = "data/DC_3hr/GSM2396856_dc_3hr_genenames.csv.gz"
mtx_fn = "data/DC_3hr/GSM2396856_dc_3hr.mtx.txt.gz"
gbc_dict_fn = "data/DC_3hr/GSM2396856_dc_3hr_cbc_gbc_dict_lenient.csv.gz"
output_dir = "output/DC_3hr_pre"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load cell barcodes and gene names
# -----------------------------------------------------------------------------
cell_df = pd.read_csv(cell_fn)
gene_df = pd.read_csv(gene_fn)

# Extract barcodes and raw gene names from column 1 (index column is 0)
cell_ids = cell_df.iloc[:, 1].astype(str).values
raw_gene_names = gene_df.iloc[:, 1].astype(str).values

# Extract clean gene symbol for pySCENIC (e.g., "ENSMUSG00000033845_Mrpl15" -> "Mrpl15")
gene_symbols = [g.split('_')[-1] if '_' in g else g for g in raw_gene_names]

# -----------------------------------------------------------------------------
# 2. Load the sparse matrix
# -----------------------------------------------------------------------------
# Raw matrix shape is (17515 genes x 33063 cells).
# Transpose to shape (33063 cells x 17515 genes) for AnnData
matrix = mmread(mtx_fn).tocsr().T

print(f"Matrix shape (cells x genes): {matrix.shape}")
print(f"Number of cell IDs: {len(cell_ids)}")
print(f"Number of gene symbols: {len(gene_symbols)}")

# -----------------------------------------------------------------------------
# 3. Create AnnData object & Format pySCENIC Attributes
# -----------------------------------------------------------------------------
adata = sc.AnnData(
    X=matrix, 
    obs=pd.DataFrame(index=cell_ids), 
    var=pd.DataFrame(index=gene_symbols)
)

# Ensure gene symbols are unique (appends -1, -2 if duplicate symbols exist)
adata.var_names_make_unique()

# *** KEY FIX FOR PYSCENIC / LOOMPY COMPATIBILITY ***
# Explicitly assign 'Gene' in var and 'CellID' in obs so loompy creates the 
# row/column attributes expected by pySCENIC.
adata.var['Gene'] = adata.var_names.values
adata.obs['CellID'] = adata.obs_names.values

# -----------------------------------------------------------------------------
# 4. Parse gbc_dict (perturbation_id -> list of cell barcodes)
# -----------------------------------------------------------------------------
cell_to_pert = {}

with gzip.open(gbc_dict_fn, "rt") as f:
    reader = csv.reader(f)
    for row in reader:
        if not row or len(row) < 2:
            continue
        pert_id = row[0].strip()
        # Combine remaining elements in row if quotes caused split, then split by comma
        barcodes_str = ",".join(row[1:])
        barcodes = [b.strip() for b in barcodes_str.split(",") if b.strip()]
        
        for bc in barcodes:
            cell_to_pert[bc] = pert_id

# Map cell barcodes in AnnData to perturbation IDs
adata.obs['perturbation'] = adata.obs.index.map(cell_to_pert)

# Unmapped cells were not infected with an sgRNA -> assign as 'control'
adata.obs['perturbation'] = adata.obs['perturbation'].fillna('control')

print("\nPerturbation cell counts (top 10):")
print(adata.obs['perturbation'].value_counts()[:10])

# -----------------------------------------------------------------------------
# 5. Export loom files for pySCENIC
# -----------------------------------------------------------------------------
# Save combined control subset (uninfected 'control' + Non-Targeting Control NTC guides)
control_mask = adata.obs['perturbation'].str.contains('control|NTC', case=False)
control_loom_path = os.path.join(output_dir, "control_cells.loom")
adata[control_mask].write_loom(control_loom_path)
print(f"\nSaved control loom to '{control_loom_path}' with {sum(control_mask)} cells.")

# Save loom files for individual perturbation conditions (with >= 30 cells)
exported_count = 0
for guide_id in adata.obs['perturbation'].unique():
    # Exclude control / NTC guides from perturbed exports
    if not any(ctrl in str(guide_id).lower() for ctrl in ['control', 'ntc']):
        subset = adata[adata.obs['perturbation'] == guide_id]
        if subset.n_obs >= 30:
            # Sanitize guide_id for filename
            clean_filename = str(guide_id).replace('/', '_').replace(' ', '_')
            out_path = os.path.join(output_dir, f"perturbed_{clean_filename}.loom")
            subset.write_loom(out_path)
            exported_count += 1

print(f"Exported {exported_count} perturbed loom files to '{output_dir}/'.")