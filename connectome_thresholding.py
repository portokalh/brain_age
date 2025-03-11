import numpy as np
import networkx as nx

def absolute_thresholding(connectomes, threshold):
    """Apply absolute thresholding: Remove edges below a fixed threshold."""
    return np.where(connectomes > threshold, connectomes, 0)

def proportional_thresholding(connectomes, proportion):
    """Keep the top proportion% of strongest connections per subject."""
    num_subjects, num_regions, _ = connectomes.shape
    thresholded = np.zeros_like(connectomes)

    for s in range(num_subjects):
        flattened = connectomes[s].flatten()
        sorted_values = np.sort(flattened)[::-1]  # Sort in descending order
        cutoff = sorted_values[int(len(sorted_values) * proportion)]
        thresholded[s] = np.where(connectomes[s] >= cutoff, connectomes[s], 0)

    return thresholded

def binarization(connectomes, threshold):
    """Binarize the connectome: 1 if above threshold, 0 otherwise."""
    return np.where(connectomes > threshold, 1, 0)

def minimum_spanning_tree(connectomes):
    """Compute the Minimum Spanning Tree (MST) for each subject."""
    num_subjects, num_regions, _ = connectomes.shape
    mst_connectomes = np.zeros_like(connectomes)

    for s in range(num_subjects):
        G = nx.Graph()
        for i in range(num_regions):
            for j in range(i + 1, num_regions):
                if connectomes[s, i, j] > 0:
                    G.add_edge(i, j, weight=connectomes[s, i, j])
        
        mst = nx.minimum_spanning_tree(G)
        for i, j, data in mst.edges(data=True):
            mst_connectomes[s, i, j] = data['weight']
            mst_connectomes[s, j, i] = data['weight']

    return mst_connectomes

if __name__ == "__main__":
    # Example usage
    n_subjects, n_regions = 100, 90
    connectomes = np.random.rand(n_subjects, n_regions, n_regions)

    abs_thresholded = absolute_thresholding(connectomes, 0.1)
    prop_thresholded = proportional_thresholding(connectomes, 0.1)
    binarized_conn = binarization(connectomes, 0.1)
    mst_conn = minimum_spanning_tree(connectomes)

    print("Original:", connectomes.shape)
    print("Absolute Thresholding:", abs_thresholded.shape)
    print("Proportional Thresholding:", prop_thresholded.shape)
    print("Binarized:", binarized_conn.shape)
    print("Minimum Spanning Tree:", mst_conn.shape)
