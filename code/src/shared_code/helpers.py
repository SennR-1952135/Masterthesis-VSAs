import torchhd
import torch
import torch.nn.functional as F
import math
from scipy import stats, integrate
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
    similarity_func = partial(similarity_func_partial, vsa_type)
    sim = similarity_func(reference_vector, vectors)
    topk_sim, topk_idx = torch.topk(sim, topk, -1)
    return topk_sim, topk_idx


def exp_acc(num_bundled, item_memory_size, dim, vsa_type):
    if vsa_type != 'BSC' and vsa_type != 'MAP':
      return 0
    if vsa_type == 'BSC':
       mu_r = 1/2
       sigma_r = math.sqrt(mu_r*(1-mu_r)/dim)
       mu_h = theoretical_similarity(num_bundled, vsa_type)
       sigma_h = math.sqrt(mu_h*(1-mu_h)/dim)
       return expected_accuracy(mu_h, sigma_h, mu_r, sigma_r, item_memory_size)
    else:
       return 0
    
def prob_corr(num_bundled, item_memory_size, dim, vsa_type):
    if vsa_type != 'BSC' and vsa_type != 'MAP':
      return 0
    if vsa_type == 'BSC':
       mu_r = 1/2
       var_r = mu_r*(1-mu_r)/dim
       mu_h = theoretical_similarity(num_bundled, vsa_type)
       var_h = mu_h*(1-mu_h)/dim
       loc = (mu_h - mu_r)
       scale = math.sqrt(var_h + var_r)
       p = 1-stats.norm.cdf(0, loc=loc, scale=scale)
       return p**num_bundled
    else:
       return 0
def expected_accuracy(mu_h, sigma_h, mu_r, sigma_r, N):
  """
  Calculates the expected accuracy (Pcorr) of retrieving the correct atomic HV.

  Args:
    mu_eta: Mean of the noise for the HV signal.
    sigma_eta: Standard deviation of the noise for the HV signal.
    mu_r: Mean of the reference HV signal.
    sigma_r: Standard deviation of the reference HV signal.
    N: Size of the item memory.

  Returns:
    The expected accuracy (Pcorr).
  """

  def integrand(x):
    return (1 / (math.sqrt(2 * math.pi) * sigma_h)) * math.exp(-(1/2)*((x - (mu_h - mu_r)) / sigma_h)**2) * (stats.norm.cdf(x, 0, sigma_r)**(N - 1))

  # Calculate the integral using numerical integration
  integral, _ = integrate.quad(integrand, -math.inf, math.inf)

  return integral

def prob_of_error(num_bundled, item_memory_size, dim, vsa_type):
    if vsa_type != 'BSC' and vsa_type != 'MAP':
      return 0

    return (1 - (1/2) * math.e**(-(num_bundled)**2/4))**(dim-1)
    # return (item_memory_size-num_bundled)* stats.norm.cdf(math.sqrt(dim/(2*num_bundled-1)), loc=0, scale=1)
    # return (1-(1-stats.norm.cdf(math.sqrt(dim/(2*num_bundled-1)), loc=0, scale=1))/2)**(item_memory_size-num_bundled)
    # return (stats.norm.cdf(math.sqrt(dim/(2*num_bundled-1)), loc=0, scale=1))**((item_memory_size-num_bundled)*num_bundled)