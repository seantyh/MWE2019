from itertools import chain
from pathlib import Path
from functools import partial
import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
from .utils import tqdm as tqdm_base
from CwnGraph import CwnBase
from .cwn_vectors import CwnVectors

tqdm = partial(tqdm_base, ascii=True)
class CwnMorphGraph:
    cache_path = Path(__file__).parent / "../data/cache_cwn_morph_graph/CwnMorphGraph.pkl"

    def __init__(self, cwn_vectors: CwnVectors=None, 
        cwn_base: CwnBase=None, debug=False):
        self.G = None
        self.debug = debug
        if cwn_vectors and cwn_base:
            cwn_vec = cwn_vectors
            self.G = nx.Graph()
            self.cwn_weight = np.mean(np.log(list(cwn_vec.freq.values())))
            self.populate_cwn_nodes(cwn_vec)
            self.populate_cwn_relations(cwn_vec, cwn_base)
            self.populate_morphologies(cwn_vec)
            self.write_cache()
        else:
            try:
                self.load_cache()                
            except:                
                print("[Error] Cannot load from cache")
                print("build CwnMorphGraph from python main.py -t morphgraph")
                raise ValueError()


    def write_cache(self):
        cache_path = CwnMorphGraph.cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_gpickle(self.G, cache_path)

    def load_cache(self):
        cache_path = CwnMorphGraph.cache_path
        self.G = nx.read_gpickle(cache_path)
        print("graph loaded from ", cache_path)

    def to_dot(self):
        self.G.graph.setdefault("node", {})
        self.G.graph["node"]["shape"] = "point"
        mg_path = Path(__file__).parent / "../data/cwn_morph_graph/cwn_morph_graph.dot"
        mg_path.parent.mkdir(parents=True, exist_ok=True)
        write_dot(self.G, mg_path)

    def populate_cwn_nodes(self, cwn_vec: CwnVectors):        
        for lemma_i, _ in tqdm(cwn_vec.itos.items(), desc='adding nodes'):
            if self.debug and lemma_i > 100: break
            self.G.add_node(lemma_i)

    def populate_cwn_relations(self, cwn_vec: CwnVectors, cwn: CwnBase):
        for lemma_i, lemma_x in tqdm(cwn_vec.itos.items(), desc='adding cwn relations'):
            if self.debug and lemma_i > 100: break
            cwn_lemmas = cwn.find_lemma(f"^{lemma_x}$")
            senses = chain.from_iterable(x.senses for x in cwn_lemmas)
            rel_senses = chain.from_iterable(sense_x.relations for sense_x in senses)
            for rel_type, rel_sense_x in rel_senses:
                if "(rev)" in rel_type: continue
                if "is_synset" in rel_type: continue
                if "generic" in rel_type: continue
                if "facet" in rel_type: continue
                if not rel_sense_x.lemmas: continue

                # avoid non-lemma nodes
                rel_lemma = rel_sense_x.lemmas[0].lemma
                if rel_lemma not in cwn_vec.stoi: continue

                # avoid self-connection in this step
                rel_lemma_id = cwn_vec.stoi[rel_lemma]
                if lemma_i == rel_lemma_id: continue

                self.G.add_edge(lemma_i, rel_lemma_id,
                        weight=self.cwn_weight,
                        reltype=rel_type)

    def populate_morphologies(self, cwn_vec: CwnVectors):
        for lemma_i, lemma_x in tqdm(cwn_vec.itos.items(), desc='adding morphologies'):
            if self.debug and lemma_i > 100: break
            if len(lemma_x) < 2: continue
            c1, c2 = lemma_x

            # avoid self-connection in this step
            if c1 == c2: continue

            c1_id = cwn_vec.stoi.get(c1)
            c2_id = cwn_vec.stoi.get(c2)
            if not (c1_id and c2_id): continue

            self.G.add_edge(c1_id, lemma_i,
                    weight=np.log(cwn_vec.freq.get(lemma_x, 1)),
                    reltype="morph")
