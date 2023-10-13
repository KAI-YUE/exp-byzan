from fedlearning.byzantine.signflipping import OmniscientSignflippingAttacker, SignflippingAttacker
from fedlearning.byzantine.alie import AlieAttacker
from fedlearning.byzantine.ipm import IPMAttacker
from fedlearning.byzantine.rop import RopAttacker
from fedlearning.byzantine.trap import ModelReplaceAttacker, NonOmniscientTrapSetter, OmniscientTrapSetter
from fedlearning.byzantine.traprop import TrapRopAttacker, TrapAlieAttacker, TrapSFAttacker
from fedlearning.byzantine.trap_maxh import TrapSetterMaxH
from fedlearning.byzantine.trap_random import RandomTrapSetter
from fedlearning.byzantine.dir_trap import DirTrapSetter
from fedlearning.byzantine.perturb import Perturb
from fedlearning.byzantine.minmax import MinMax

from fedlearning import LocalUpdater

attacker_registry = {
    "signflipping":             SignflippingAttacker,
    "omniscient_signflipping":  OmniscientSignflippingAttacker,

    "ipm":                      IPMAttacker,
    "alie":                     AlieAttacker,
    "rop":                      RopAttacker, 

    "minmax":                   MinMax,

    "model_replace":            ModelReplaceAttacker,
    
    "omniscient_trapsetter":    OmniscientTrapSetter,
    "nonomniscient_trapsetter": NonOmniscientTrapSetter,

    "max_h":                    TrapSetterMaxH,
    "traprop":                  TrapRopAttacker,
    "trapalie":                 TrapAlieAttacker,
    "trapsf":                   TrapSFAttacker,
    "trap_random":              RandomTrapSetter,
    "dir_trap":                 DirTrapSetter,

    "benign":                   LocalUpdater,
    "perturb":                  Perturb,
}

def init_attacker(config):
    attacker = attacker_registry[config.attacker_model]

    return attacker