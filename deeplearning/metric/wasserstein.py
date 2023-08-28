from scipy.stats import wasserstein_distance

def wasserstein_metric(a, b):
    return wasserstein_distance(a, b)