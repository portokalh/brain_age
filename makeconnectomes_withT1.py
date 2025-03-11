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
parser.add_argument("--t1", type=str, required=True, help="Path to T1-weighted MRI NIfTI file")
parser.add_argument("--atlas", type=str, default="destrieux", help="Atlas to use (default: destrieux)")
parser.add_argument("--method", type=str, default="ants", choices=["ants", "flirt"], help="Registration method (ANTs or FLIRT)")
parser.add_argument("--output", type=str, required=True, help="Output directory for connectomes")
args = parser.parse_args()

# ------------------------
# 2. Register T1W to DWI Space
# ------------------------
t1_registered = os.path.join(args.output, "t1_registered.nii.gz")
t1_brain = os.path.join(args.output, "t1_brain.nii.gz")

# Skull-strip T1W
subprocess.run(["bet", args.t1, t1_brain])

# Register T1 to DWI
if args.method == "flirt":
    subprocess.run(["flirt", "-in", t1_brain, "-ref", args.dwi, "-out", t1_registered, "-omat", os.path.join(args.output, "t1_to_dwi.mat")])
elif args.method == "ants":
    subprocess.run(["antsRegistrationSyNQuick.sh", "-d", "3", "-f", args.dwi, "-m", t1_brain, "-o", os.path.join(args.output, "ants_t1_to_dwi")])
    t1_registered = os.path.join(args.output, "ants_t1_to_dwiWarped.nii.gz")

# ------------------------
# 3. Apply Atlas in T1W Space
# ------------------------
atlas_path = datasets.fetch_atlas_destrieux_2009().maps
atlas_registered = os.path.join(args.output, "atlas_registered.nii.gz")

if args.method == "flirt":
    subprocess.run(["flirt", "-in", atlas_path, "-ref", t1_registered, "-out", atlas_registered, "-applyxfm", "-init", os.path.join(args.output, "t1_to_dwi.mat"), "-interp", "nearestneighbour"])
elif args.method == "ants":
    subprocess.run(["antsApplyTransforms", "-d", "3", "-i", atlas_path, "-r", t1_registered, "-o", atlas_registered, "-n", "NearestNeighbor", "-t", os.path.join(args.output, "ants_t1_to_dwiWarped.nii.gz")])

# ------------------------
# 4. Generate Connectome Matrix
# ------------------------
connectome_file = os.path.join(args.output, "connectome.csv")
subprocess.run(["tck2connectome", os.path.join(args.output, "tracks.tck"), atlas_registered, connectome_file, "-scale_invnodevol", "-stat_edge", "mean", "-symmetric"])

# ------------------------
# 5. Load and Process Connectome
# ------------------------
connectome = np.loadtxt(connectome_file, delimiter=',')
num_nodes = connectome.shape[0]

# Save as CSV
connectome_df = pd.DataFrame(connectome)
csv_path = os.path.join(args.output, "connectome_matrix.csv")
connectome_df.to_csv(csv_path, index=False, header=False)
print(f"Saved connectome matrix to {csv_path}")

# ------------------------
# 6. Convert to Graph Representation
# ------------------------
G = nx.from_numpy_matrix(connectome)
edge_list_path = os.path.join(args.output, "connectome_edgelist.txt")
nx.write_edgelist(G, edge_list_path, data=['weight'])
print(f"Saved edge list to {edge_list_path}")

# ------------------------
# 7. Save Graph as Adjacency Matrix
# ------------------------
adjacency_path = os.path.join(args.output, "connectome_adj.npz")
sp.save_npz(adjacency_path, sp.csr_matrix(connectome))
print(f"Saved adjacency matrix to {adjacency_path}")
