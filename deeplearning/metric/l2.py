import numpy as np

def l2_metric(p, q):
    """Calculate L2 distance."""
    return np.linalg.norm(p - q)