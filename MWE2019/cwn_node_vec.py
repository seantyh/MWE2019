import pickle
from pathlib import Path
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from .cwn_vectors import CwnVectors
from .cwn_morph_graph import CwnMorphGraph

class CwnNodeVec:
    cache_fname = "cwn_node_vec_{suffix}.pkl"
    cache_dir = Path(__file__).parent / "../data/cache_cwn_node_vec/"
    def __init__(self, name, **kwargs):
        self.name = name
        self.embed = None
        self.itos = None
        self.stoi = None
        try:
            self.load_cache()
        except FileNotFoundError:            
            cwn_vec = CwnVectors()
            cwn_mg = CwnMorphGraph()
            self.build_node_vec(cwn_mg, cwn_vec, **kwargs)
            self.write_cache()

    def write_cache(self):
        cache_dir = CwnNodeVec.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / CwnNodeVec.cache_fname.format(suffix=self.name)
        with open(cache_path, "wb") as fout:
            data = (self.embed, self.stoi, self.itos)
            pickle.dump(data, fout)
        print("CwnNodeVec write to: ", cache_path)

    def load_cache(self):
        cache_dir = CwnNodeVec.cache_dir
        cache_path = cache_dir / CwnNodeVec.cache_fname.format(suffix=self.name)
        with open(cache_path, "rb") as fin:
            self.embed, self.stoi, self.itos = pickle.load(fin)
    
    def build_node_vec(self, cwn_mg: CwnMorphGraph, cwn_vec: CwnVectors, **kwargs):
        n_dim = kwargs.get("dimensions", 100)
        walk_length = kwargs.get("walk_length", 10)
        num_walks = kwargs.get("num_walks", 20)
        p = kwargs.get("p", 1)
        q = kwargs.get("q", 1)
        node2vec = Node2Vec(cwn_mg.G, 
                    dimensions=n_dim, walk_length=walk_length, num_walks=num_walks, 
                    p=p, q=q)
        sg_model = node2vec.fit()        
        self.embed = sg_model.wv
        self.itos = cwn_vec.itos
        self.stoi = cwn_vec.stoi

    def node_most_similar(self, word:str, charonly=True, topn=5):    
        itos = self.itos
        stoi = self.stoi
        wv = self.embed
        candids = wv.most_similar(str(stoi[word]), topn=100)
        ret = []
        for candid_i, val in candids:        
            candid_s = itos[int(candid_i)]
            if charonly and len(candid_s) > 1:
                continue
            ret.append((candid_s, val))
            if len(ret) >= topn:
                break
        return ret
    
    def prob_x2(self, x1):
        wv = self.embed

        # only restrict to single-word (character)
        charidxs = [i for s, i in self.stoi.items() if len(s) == 1]

        x1_idx = self.stoi[x1]    
        other_nodes = [str(x) for x in charidxs if x != x1_idx]
        # distances function already convert cosine similarity to distance measure
        # with $1 - cos_similarity$, c.f.
        # https://github.com/RaRe-Technologies/gensim/blob/bd199aa0382bbbabcf7862ca80f65d7484a450cb/gensim/models/keyedvectors.py#L918

        scores = 1- wv.distances(str(self.stoi[x1]), other_nodes)     
        
        # convert to probability with softmax function
        Z = np.sum(np.exp(scores))
        probs = np.exp(scores) / Z        
        ret = {self.itos[int(other_nodes[i])]: p for i, p in enumerate(probs)}
        return ret