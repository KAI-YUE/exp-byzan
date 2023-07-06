from fedlearning.byzantine.signflipping import OmniscientSignflippingAttacker, SignflippingAttacker
from fedlearning.byzantine.fang import FangattackAdversary

attacker_registry = {
    "signflipping":            SignflippingAttacker,
    "omniscient_signflipping": OmniscientSignflippingAttacker,

    "fang":                    FangattackAdversary,
}

def init_attacker(config):
    attacker = attacker_registry[config.attacker_model]

    return attacker