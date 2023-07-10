from .mean import Mean, BenignFedAvg
from .median import Median
from .krum import Multikrum
from .trimmedmean import TrimmedMean
from .centerclipping import CenteredClipping
from .vote import Vote

aggregator_registry = {
    "mean":             Mean,
    "median":           Median,
    "trimmed_mean":     TrimmedMean,
    "centeredclipping":  CenteredClipping,

    "krum":             Multikrum,
    "vote":             Vote
}
