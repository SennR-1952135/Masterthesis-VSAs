import torchhd
import torch.nn.functional as F
import math
from scipy import stats


def norm_hamming_similarity(x, y):
    normalisation_factor = x.shape[1] if x.ndim == 2 else x.shape[0]
    sim = torchhd.hamming_similarity(x, y)
    norm_sim = sim / normalisation_factor
    return norm_sim

def similarity_func_partial(vsa_type, x, y):
    if vsa_type == 'BSC':
        return norm_hamming_similarity(x, y)
    else:
        return torchhd.cosine_similarity(x, y)
    
def normalised_bundle_partial(vsa_type, x, y):
    bundle_vec = x.bundle(y)
    return normalise_partial(vsa_type, bundle_vec)
    
def normalise_partial(vsa_type, x):
    if vsa_type == 'MAP':
        return x.clipping(1)
    elif vsa_type == 'HRR':
        return F.normalize(x, p=2, dim=-1)
    else:
        return x
    
def theoretical_similarity(amount_bundled, vsa_type):
    if vsa_type != 'BSC':
      return 0
    amount_bundled = amount_bundled + 1 if amount_bundled % 2 == 0 else amount_bundled # +1 because we need an odd amount of elements
    expected_similarity = 1/2 + (math.comb(amount_bundled-1, int((amount_bundled-1)/2)) / 2**amount_bundled)
    return expected_similarity

def similarity_cutoff(bundle_size, dim, certainty:float = 0.9):
  exp_sim = theoretical_similarity(bundle_size, 'BSC')
  var = exp_sim * (1-exp_sim) * (1/dim)
  cutoff = exp_sim + stats.norm.ppf(1-certainty) * math.sqrt(var)
  return cutoff