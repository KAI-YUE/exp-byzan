# My Libraries
from deeplearning.metric.corr import corr_metric
from deeplearning.metric.wasserstein import wasserstein_metric
from deeplearning.metric.kl import symmetric_kl_metric
from deeplearning.metric.l2 import l2_metric
from deeplearning.metric.cosine import cosine_metric

metric_registry = {
    "wasserstein":      wasserstein_metric,
    # "corr":             corr_metric,
    "kl":               symmetric_kl_metric,
    "l2":               l2_metric,
    "cosine":           cosine_metric
}
