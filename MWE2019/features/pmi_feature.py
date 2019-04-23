from typing import List, Dict
from functools import partial
import numpy as np
from ..materials import Materials

def make_pmi_features(materials: Materials):
    uniFreq = materials.chFreq
    logZ = np.log(np.sum(list(uniFreq.values())))
    _compute_pmi = partial(compute_pmi, uniFreq=uniFreq, 
                    ngramFreq=materials.ngFreq, logZ=logZ)
    
    qie_pmi = [_compute_pmi(x) for x in materials.qies]
    idiom_pmi = [_compute_pmi(x) for x in materials.idioms]
    return qie_pmi + idiom_pmi

def compute_pmi(chars:List[str], uniFreq, ngramFreq, logZ):
    ch_freqs = [uniFreq.get(x, 1) for x in chars]
    ng_freq = ngramFreq.get(chars, 1)
    denom = np.sum(np.log(ch_freqs))
    pmi = np.log(ng_freq) - denom + logZ * 3
    return pmi
