from fedlearning.byzantine.signflipping import OmniscientSignflippingAttacker, SignflippingAttacker
from fedlearning.byzantine.alie import AlieAttacker
from fedlearning.byzantine.ipm import IPMAttacker

attacker_registry = {
    "signflipping":            SignflippingAttacker,
    "omniscient_signflipping": OmniscientSignflippingAttacker,

    "ipm":                     IPMAttacker,
    "alie":                    AlieAttacker,
    # "fang":                    FangattackAdversary,
}

def init_attacker(config):
    attacker = attacker_registry[config.attacker_model]

    return attacker