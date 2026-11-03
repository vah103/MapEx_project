"""Stand-in for pc_model.p_free until topo-l1 delivers it: p = 1[mean < 0.5]."""
import numpy as np


def p_free(obs, mean, var):
    return (np.asarray(mean) < 0.5).astype(np.float32)
