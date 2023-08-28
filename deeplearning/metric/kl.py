from scipy.stats import entropy

def symmetric_kl_metric(p, q):
    """Calculate symmetric KL Divergence."""
    return 0.5 * (entropy(p, q) + entropy(q, p))