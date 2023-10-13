from .mean import Mean, BenignFedAvg
from .median import Median
from .krum import Multikrum
from .trimmedmean import TrimmedMean
from .centerclipping import CenteredClipping
from .vote import Vote
from .momentum import MeanMomentum
from .signguard import SignGuard
from .dnc import DnC

aggregator_registry = {
    "mean":             Mean,
    "median":           Median,
    "trimmed_mean":     TrimmedMean,
    "centeredclipping":  CenteredClipping,

    "krum":             Multikrum,
    "vote":             Vote,

    "mean_momentum":    MeanMomentum,
    "signguard":        SignGuard,

    "dnc":              DnC,
}
