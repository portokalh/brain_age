import os
import numpy as np
import subprocess
import argparse
import scipy.sparse as sp
import pandas as pd
import networkx as nx
from nilearn import datasets

# ------------------------
# 1. Command-line Arguments
# ------------------------
parser = argparse.ArgumentParser(description="Generate Connectomes from Diffusion MRI using Destrieux Atlas")
parser.add_argument("--dwi", type=str, required=True, help="Path to 4D Diffusion MRI NIfTI file")
parser.add_argument("--bvec", type=str, required=True, help="Path to bvec file")
parser.add_argument("--bval", type=str, required=True, help="Path to bval file")
parser.add_argument("--atlas", type=str, default="destrieux", help="Atlas to use (default: destrieux)")
parser.add_argument("--method", type=str, default="ants", choices=["ants", "flirt"], help="Registration method (ANTs or FLIRT)")
parser.add_argument("--output", type=str, required=True, help="Output directory for connectomes")
args = parser.parse_args()

# ------------------------
# 2. Generate Connectome Matrix
# ------------------------
connectome_file = os.path.join(args.output, "connectome.csv")
subprocess.run(["tck2connectome", os.path.join(args.output, "tracks.tck"), 
                os.path.join(args.output, "destrieux_registered.nii.gz"), 
                connectome_file, "-scale_invnodevol", "-stat_edge", "mean", "-symmetric"])

# ------------------------
# 3. Load and Process Connectome
# ------------------------
connectome = np.loadtxt(connectome_file, delimiter=',')
num_nodes = connectome.shape[0]

# Save as CSV
connectome_df = pd.DataFrame(connectome)
csv_path = os.path.join(args.output, "connectome_matrix.csv")
connectome_df.to_csv(csv_path, index=False, header=False)
print(f"Saved connectome matrix to {csv_path}")

# ------------------------
# 4. Convert to Graph Representation
# ------------------------
G = nx.from_numpy_matrix(connectome)
edge_list_path = os.path.join(args.output, "connectome_edgelist.txt")
nx.write_edgelist(G, edge_list_path, data=['weight'])
print(f"Saved edge list to {edge_list_path}")

# ------------------------
# 5. Save Graph as Adjacency Matrix
# ------------------------
adjacency_path = os.path.join(args.output, "connectome_adj.npz")
sp.save_npz(adjacency_path, sp.csr_matrix(connectome))
print(f"Saved adjacency matrix to {adjacency_path}")
