import numpy as np
from typing import Dict, List
from functools import partial
from ..materials.materials import Materials, Character, MVector
from ..utils import tqdm
from .common import cosine_similarities

def make_compoundness_feature(materials: Materials):
    stoi, mvecs = get_M_vectors(materials)
    _compute = partial(compute_compoundness, charM=materials.charM, 
                    mvecs=mvecs, stoi=stoi)
    qie_comp = [_compute(x) for x in tqdm(materials.qies, desc='qies')]
    idiom_comp = [_compute(x) for x in tqdm(materials.idioms, desc='idioms')]
    return qie_comp + idiom_comp

def debug_prob_x2(x1, materials):
    stoi, mvecs = get_M_vectors(materials)    
    try:
        x1_vec = materials.charM[x1]
    except KeyError:
        x1_vec = get_random_M()
    prob = prob_x2(x1_vec, mvecs)
    return {ch: prob[idx] for ch, idx in stoi.items()}

def compute_compoundness(chars: List[str], 
        charM: Dict[Character, MVector],
        mvecs: List[MVector], 
        stoi: Dict[Character, int]):    
    logP = 0
    for x1, x2 in zip(chars, chars[1:]):
        try:
            x1_vec = charM[x1]
        except KeyError:
            x1_vec = get_random_M()
        
        pvec = prob_x2(x1_vec, mvecs)
        if x2 in stoi:
            x2_prob = pvec[stoi[x2]]
        else:
            x2_prob = np.mean(pvec)

        logP += np.log(x2_prob)
    return logP

def get_M_vectors(materials: Materials):
    stoi = {s: i for i, s in enumerate(materials.charM.keys())}
    m_vecs = np.vstack(list(materials.charM.values()))
    return stoi, m_vecs

def get_random_M(seed=13253):
    random_m = np.random.randn(100)
    return random_m

def prob_x2(x1vec: np.array, m_vectors: np.array):
    scores = cosine_similarities(x1vec, m_vectors)

    # convert to probability with softmax function
    Z = np.sum(np.exp(scores))
    probs = np.exp(scores) / Z
    return probs