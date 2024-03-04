import torchhd
import torch
import torch.nn.functional as F
import math
from scipy import stats
from functools import partial

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

def similarity_cutoff(bundle_size, dim, certainty:float = 0.9, vsa_type:str = 'BSC'):
  exp_sim = theoretical_similarity(bundle_size, vsa_type)
  var = exp_sim * (1-exp_sim) * (1/dim)
  cutoff = exp_sim + stats.norm.ppf(1-certainty) * math.sqrt(var)
  return cutoff

def top_k_vectors(reference_vector, vectors, topk=1, vsa_type='BSC'):
    r"""Return top k most similar vectors to reference_vector
    Args:
      reference_vector: vector to compare to
      vectors: list of vectors to compare against
      topk: amount of most similar vectors to return
      vsa_type: type of VSA used
    Returns:
      topk_sim: list of top k similarities
      topk_idx: list of top k indices
    """
    # remove reference_vector from vectors if it is in there
    vectors[vectors != reference_vector]
    similarity_func = partial(similarity_func_partial, vsa_type)
    sim = similarity_func(reference_vector, vectors)
    topk_sim, topk_idx = torch.topk(sim, topk, -1)
    return topk_sim, topk_idx