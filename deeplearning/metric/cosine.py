import numpy as np

# def cosine_metric(p, q):
#     return np.sum(p*q)/(np.linalg.norm(p)*np.linalg.norm(q))

def cosine_metric(p, q):
    return 1 - np.sum(p*q)/(np.linalg.norm(p)*np.linalg.norm(q))