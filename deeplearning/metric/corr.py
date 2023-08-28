import scipy.stats as stats

def corr_metric(p, q):
    correlation = stats.pearsonr(p, q)[0]
    return correlation