from .mean import Mean, BenignFedAvg
from .median import Median
from .krum import Multikrum

aggregator_registry = {
    "mean":     Mean,
    "median":   Median,

    "krum":     Multikrum,
}
