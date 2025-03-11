import numpy as np
import pandas as pd
from scipy.stats import zscore

def zscore_normalization(connectomes):
    """Normalize each edge across subjects using Z-score normalization."""
    mean = np.mean(connectomes, axis=0)
    std = np.std(connectomes, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    return (connectomes - mean) / std

def global_mean_normalization(connectomes):
    """Normalize each connectome by its global mean connectivity."""
    mean_per_subject = np.mean(connectomes, axis=(1,2), keepdims=True)
    return connectomes / mean_per_subject

def row_wise_normalization(connectomes):
    """Normalize each row in the connectivity matrix so that row sums equal 1."""
    row_sums = np.sum(connectomes, axis=2, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    return connectomes / row_sums

def log_zscore_normalization(connectomes):
    """Apply log transformation followed by Z-score normalization."""
    log_transformed = np.log1p(connectomes)  # Log(1 + x) transformation
    return zscore_normalization(log_transformed)

if __name__ == "__main__":
    # Example usage
    n_subjects, n_regions = 100, 90
    connectomes = np.random.rand(n_subjects, n_regions, n_regions)

    zscore_conn = zscore_normalization(connectomes)
    global_mean_conn = global_mean_normalization(connectomes)
    row_normalized_conn = row_wise_normalization(connectomes)
    log_zscore_conn = log_zscore_normalization(connectomes)

    print("Original:", connectomes.shape)
    print("Z-score Normalized:", zscore_conn.shape)
    print("Global Mean Normalized:", global_mean_conn.shape)
    print("Row-wise Normalized:", row_normalized_conn.shape)
    print("Log + Z-score Normalized:", log_zscore_conn.shape)
