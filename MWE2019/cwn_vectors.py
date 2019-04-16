import pickle
from pathlib import Path
from typing import List, Dict, Iterable
from torchtext.vocab import Vectors
import numpy as np

CwnLemmas = List[str]

class CwnVectors:
    cache_path = Path(__file__).parent / "../data/cache_cwn_vectors/cwn_vectors.pkl"
        
    def from_cache(self):
        cache_path = CwnVectors.cache_path
        inst = None
        with open(cache_path, "rb") as fin:
            print("load from cache: ", cache_path.resolve())
            data = pickle.load(fin)            
            (self.stoi, self.itos, 
                self.dim, self.vectors, self.freq) = data
        return inst


    def __init__(self, cwn_lemmas: CwnLemmas=None,
        pt_vectors: Vectors=None, word_freq: Dict[str,int]=None):
        """CwnVectors

        parameters
        -----------
        cwn_lemmas: List[str], lemmas from CWN data
        pt_vectors: Vectors, a pretrained word embedding,
            presumbaly loaded from torchtext.Vectors
        word_freq: Dict[int, str], a dictionary carrying
            word frequency information
        """

        self.itos = None
        self.stoi = None
        self.vectors = None
        self.dim = None

        if cwn_lemmas and pt_vectors and word_freq:
            self.build_vectors(cwn_lemmas, pt_vectors, word_freq)
            self.write_cache()
        else:
            try:
                self.from_cache()
            except:
                pass

    def build_vectors(self, cwn_lemmas, pt_vectors, word_freq):
        lemmas = set(cwn_lemmas.copy())
        lemmas.intersection_update(pt_vectors.stoi.keys())
        lemmas.intersection_update(word_freq.keys())

        lemmas = list(lemmas)
        lemmas.sort(key=lambda x: (len(x), x))
        vectors = np.zeros((len(lemmas), pt_vectors.dim))
        stoi = {}
        itos = {}
        freq = {}
        for lemma_i, lemma_x in enumerate(lemmas):
            pt_vec_idx = pt_vectors.stoi.get(lemma_x)
            freq_x = word_freq.get(lemma_x)
            if not pt_vec_idx or not freq_x: continue
            vectors[lemma_i, :] = pt_vectors.vectors[pt_vec_idx, :]
            stoi[lemma_x] = lemma_i
            itos[lemma_i] = lemma_x
            freq[lemma_x] = freq_x
        self.vectors = vectors
        self.stoi = stoi
        self.itos = itos
        self.dim = pt_vectors.dim
        self.freq = freq

    def write_cache(self):
        cache_path = CwnVectors.cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fout:
            pickle.dump((self.stoi, self.itos, 
                self.dim, self.vectors, self.freq), fout)