import torchhd
import torch.nn.functional as F

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
    if vsa_type == 'MAP':
        return bundle_vec.clipping(1)
    elif vsa_type == 'HRR':
        return F.normalize(bundle_vec, p=2, dim=-1)
    else:
        return bundle_vec