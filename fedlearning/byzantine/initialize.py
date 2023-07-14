from fedlearning.byzantine.signflipping import OmniscientSignflippingAttacker, SignflippingAttacker
from fedlearning.byzantine.alie import AlieAttacker
from fedlearning.byzantine.ipm import IPMAttacker
from fedlearning.byzantine.rop import RopAttacker

attacker_registry = {
    "signflipping":            SignflippingAttacker,
    "omniscient_signflipping": OmniscientSignflippingAttacker,

    "ipm":                     IPMAttacker,
    "alie":                    AlieAttacker,
    "rop":                     RopAttacker
}

def init_attacker(config):
    attacker = attacker_registry[config.attacker_model]

    return attacker