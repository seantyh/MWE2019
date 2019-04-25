import pickle
from itertools import chain
from textwrap import dedent
from typing import List, Iterable, Dict, Union
import numpy as np
import pandas as pd

from ..utils import get_cache_path, install_data_cache
from ..cwn_node_vec import CwnNodeVec

QIE = str
Idiom = str
Freq = int
NGram = Union[QIE, Idiom]
Character = str
MVector = np.array
SVector = np.array


class Materials:
    def __init__(self): 
        ## initialize fields       
        self.qies: List[QIE] = []
        self.idioms: List[Idiom] = []
        self.ngFreq: Dict[NGram, Freq] = {}
        self.chFreq: Dict[Character, Freq] = {}

        self.charM: Dict[Character, MVector] = {}
        self.charS: Dict[Character, SVector] = {}
        self.ngramS: Dict[NGram, SVector] = {}
        self.variations: Dict[NGram, int] = {}

        self.proper_qie = set()

        ## load data        
        self.load_qies()
        self.load_idioms()
        self.load_unigram()        
        self.preprocess_materials()

        self.load_character_Svector()
        self.load_character_Mvector()
        self.load_ngram_Svector()

        self.load_variations()
        self.load_proper()
    
    def describe(self):
        mdtext = f"""
        ## Materials of MWE2019
        ### NGram list ({len(self.ngFreq)} ngrams)
          * QIE (`qies`): {len(self.qies)} QIEs, e.g. {", ".join(self.qies[:5])} ...  
            These QIEs are selected with high PMI (top 25%) and low enclosing frequency (top 25%), 
            and they are not idioms as defined in MOE.
          * Idiom (`idioms`): {len(self.idioms)} idioms, e.g. {", ".join(self.idioms[:5])}, ...
          * These ngrams (QIEs and idioms) are all occured 5 or more times in 1.3 billion corpus.        

        ### Characters 
          * There are {len(self.chFreq)} used in all ngrams
          * CWN contains information of {len(self.charM)} character
          * Number of characters both in CWN and used in NGrams are 
            `{len(set(self.chFreq.keys()).intersection(self.charM.keys()))}`, and
            `{len(set(self.chFreq.keys()).difference(self.charM.keys()))}` of them are only presented in 
            ngrams but not in CWN.
          * Some characters do not have S-vectors due to data problem, only 1983 characters have 
            valid senses and S-vectors          

        ### Frequency
          * NGram frequency (`ngFreq`): Frequency of QIEs and idioms in 1.3 billion corpus
          * character frequency (`chFreq`): Frequency of those `{len(self.chFreq)}` character, 
            as used by the ngrams, in the same corpus.
        
        ### Vector representation
          * Character M-Vector (Morphological Vector)  
            computed from CWN networks with node2vec (`CwnNodeVec`), each of {len(self.charM)} character 
            was mapped to a vector of length {next(iter(self.charM.values())).shape[0]}.
          * Character S-Vector (Sense Vector)  
            computed from the example sentences in CWN senses, as described in GWA2019 paper. Each of 
            {len(self.charM)} character was mapped to a vector of length {self.charS["我"]["05238701"].shape[0]}
          * NGrams S-vector
            computed from the sentences extracted from corpus. Each of {len(self.ngramS)} ngram was mapped
            to a vector of length {self.ngramS["主辦單位"].shape[0]}
        """

        return dedent(mdtext)

    def describe_cwn_char_diff(self):
        cwn_chars = set(self.charM.keys())
        X = sum([freq for ch, freq in self.chFreq.items() if ch not in cwn_chars])
        return X/sum(self.chFreq.values())

    def load_qies(self):
        qie_cache_path = get_cache_path('qie_list', 'qie_list_full.csv')
        qie_list = pd.read_csv(qie_cache_path, index_col='ngram')
        self.qies = qie_list.index.values.tolist()
        qie_freq = dict(zip(qie_list.index.values, qie_list.exact_freq))
        self.ngFreq.update(qie_freq)
    
    def load_idioms(self):
        idiom_path = get_cache_path('qie_list', 'idiom_exact_frequency.csv')
        idiom_list = pd.read_csv(idiom_path, index_col=0)
        self.idioms = idiom_list.index.values.tolist()
        idiom_freq = dict(zip(idiom_list.index.values, idiom_list.exact_freq))
        self.ngFreq.update(idiom_freq)
        
    def load_unigram(self):
        unigram_path = get_cache_path('cache_corpus_unigram', 'unigram.pkl')
        with open(unigram_path, 'rb') as fin:
            unigram = pickle.load(fin)
        
        self.chFreq.update(unigram)
    
    def preprocess_materials(self):
        # eliminate NGram frequency < 50 and those not composing with 4 characters
        ng_to_remove = [x for x, f in self.ngFreq.items() if f < 50 or len(x) != 4]
        n_qie_removed = 0
        n_idiom_removed = 0
        for ng_x in ng_to_remove:
            try:
                self.qies.remove(ng_x)
                n_qie_removed += 1
            except:
                pass

            try:
                self.idioms.remove(ng_x)
                n_idiom_removed += 1
            except:
                pass
            
            del self.ngFreq[ng_x]
        print(f"Remove NGram frequency < 50: {len(ng_to_remove)} removed")
        print(f"QIE removed: {n_qie_removed}")
        print(f"Idiom removed: {n_idiom_removed}")

        # eliminate all characters not in ngrams
        uniq_chars = set(chain.from_iterable(self.ngFreq.keys()))
        n_char_before = len(self.chFreq)
        print(f"Character count before removal: {n_char_before}")
        self.chFreq = {ch: freq for ch, freq in self.chFreq.items() if ch in uniq_chars }
        n_char_after = len(self.chFreq)
        print(f"Character count after removal: {n_char_after}")
        print(f"Remove character not in ngrams: {n_char_before - n_char_after} removed")

    def load_character_Mvector(self):
        cwn_nv_homo = CwnNodeVec(name='homophily', dimensions=100, walk_length=10, num_walks=20, p=2, q=0.5)
        # note cwn_nv.embed is a gensim.KeyedVector
        for s, idx in cwn_nv_homo.stoi.items():
            # skip characters not in materials
            if s not in self.chFreq: 
                continue
            self.charM[s] = cwn_nv_homo.embed.get_vector(str(idx))
    
    def load_character_Svector(self):
        char_senses_path = get_cache_path('sense_vectors', 'char_senses.pkl')
        with open(char_senses_path, "rb") as fin:
            char_senses = pickle.load(fin)
        
        # skip characters not in materials
        char_senses_iter = filter(lambda item: item[0] in self.chFreq, char_senses.items())
        for ch, sense_data in char_senses_iter:
            npvec_data = {}          
            for sid, svec in sense_data.items():
                try:
                    npvec = svec.numpy()
                except:
                    npvec = svec
                npvec_data[sid] = npvec
            self.charS[ch] = npvec_data
    
    def load_ngram_Svector(self):
        qie_vectors_path = get_cache_path("sense_vectors", "qie_vectors.pkl")
        with open(qie_vectors_path, "rb") as fin:
            qie_vectors = pickle.load(fin)
        
        # skip characters not in materials
        qie_vectors_iter = filter(lambda item: item[0] in self.ngFreq, qie_vectors.items())
        self.ngramS.update(qie_vectors_iter)
        
    def load_variations(self):        
        variation_path = get_cache_path('cache_ngrams_list', "ngrams_vars_samples.csv")
        print("load variations from ", variation_path)
        ng_list = pd.read_csv(variation_path, index_col='ngram')
        for ridx, row in ng_list.iterrows():              
            self.variations[ridx] = row["var"]
        
    def load_proper(self):
        qie_proper_path = get_cache_path('qie_list', 'qie_proper.csv')
        print("Load QIE proper: ", qie_proper_path)
        proper_list = pd.read_csv(qie_proper_path)
        self.proper_qie = set(proper_list.ngram)

