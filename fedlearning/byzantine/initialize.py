from fedlearning.byzantine.signflipping import SignflippingAttacker

attacker_registry = {
    "signflipping":            SignflippingAttacker,
}

def init_attacker(config):
    attacker = attacker_registry[config.attacker_model]

    return attacker