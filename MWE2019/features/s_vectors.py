import numpy as np
from functools import partial
from typing import List, Dict
from ..utils import tqdm
from .m_vectors import prob_x2
from ..materials.materials import Materials, Character, MVector, SVector, NGram

# PVector = SVector * W
PVector = np.array
SenseId = str
SensesPMap = Dict[SenseId, PVector]

def make_compositioness_feature(materials: Materials):
    W = get_linear_W(materials)
    _compute = partial(compute_compositioness, matW=W, 
                charS=materials.charS, ngramS=materials.ngramS)
    qie_comp = [_compute(x) for x in tqdm(materials.qies, desc='qies')]
    idiom_comp = [_compute(x) for x in tqdm(materials.idioms, desc='idioms')]
    return qie_comp + idiom_comp

def build_senses_sequences(chars: List[str], matW: np.array, charS: Dict[Character, SVector]):
    senses_list = []
    for ch in chars:
        sense_data = charS.get(ch)
        ch_senses = {}
        if not sense_data or len(sense_data) == 0:
            randS = get_random_S()
            ch_senses['random'] = np.dot(randS, matW)
        else:
            ch_senses = {
                sid: np.dot(svec, matW)
                for sid, svec in sense_data.items()
            }
        senses_list.append(ch_senses)
    return senses_list

def compute_compositioness(chars:List[str], matW: np.array, 
        charS: Dict[Character, SVector], 
        ngramS: Dict[NGram, SVector]):

    senses_list = build_senses_sequences(chars, matW, charS)
    
    # decode senses_list
    SS_distances = []
    sequences = beam_search_decoder(senses_list, k=1)
    for seq, prob in sequences:
        try:
            qie_svec = ngramS.get[chars]                
        except:
            qie_svec = get_random_S()

        pvectors = []
        for ch, sid in zip(chars, seq):
            try:
                svec_x = charS[ch][sid]                
            except:
                svec_x = get_random_S()
            pvectors.append(svec_x)        
        pvectors = np.array(pvectors)
        diff_vecs = pvectors - qie_svec
        diff_SS = np.sqrt(np.diag(np.dot(diff_vecs, diff_vecs.transpose())))        
        scaled_SS = np.sum(diff_SS)/qie_svec.shape[0]        
        SS_distances.append(scaled_SS)    
    compositioness = np.mean(SS_distances)

    return compositioness


def get_random_S(seed=13253):
    random_s = np.random.randn(3072)
    return random_s

def get_s1_vectors(materials: Materials):
    s1map = {}
    for charac, sense_data in materials.charS.items():
        try:
            sid1, svec1 = next(iter(sense_data.items()))
            s1map[charac] = svec1
        except:
            continue
    return s1map

def get_linear_W(materials: Materials):
    mvectors = materials.charM
    s1vectors = get_s1_vectors(materials)

    mvecs = []
    svecs = []
    for ch in s1vectors:
        mvecs.append(mvectors[ch])
        svecs.append(s1vectors[ch])
    matM = np.array(mvecs)
    matS = np.array(svecs)
    StS = np.dot(matS.transpose(), matS)
    linW = np.dot(np.dot(np.linalg.inv(StS), matS.transpose()), matM)

    return linW


# beam search
# adapted from https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/
def beam_search_decoder(senses_seq: List[SensesPMap], k):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for t, sense_x in enumerate(senses_seq):
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            try:
                seq, score = sequences[i]
                pvec_i = senses_seq[t-1][seq[-1]]
                sense_stoi, sense_pvecs = get_pvec_matrix(sense_x)
                score_vec = prob_x2(pvec_i, sense_pvecs)
                score_map = {sid: score_vec[idx] for sid, idx in sense_stoi.items()}
            except IndexError:
                # it's the first timestamp
                # assume uniform prior
                score_map = {sid:1/len(sense_x) for sid in sense_x.keys()}

            for sense_id in sense_x.keys():
                p_score = score_map[sense_id]
                candidate = [seq + [sense_id], score * p_score]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1], reverse=True)
        # select k best
        sequences = ordered[:k]
    return sequences

def get_pvec_matrix(sense_pmap: SensesPMap):
    stoi = {s: i for i, s in enumerate(sense_pmap.keys())}
    p_vecs = np.vstack(list(sense_pmap.values()))
    return stoi, p_vecs