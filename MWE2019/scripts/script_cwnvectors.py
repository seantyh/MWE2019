import pickle
import gzip
import json
from torchtext.vocab import Vectors
from CwnGraph import CwnBase
from pathlib import Path
from ..utils import tqdm
from ..cwn_vectors import CwnVectors

def script_cwnvectors(**kwargs):
    # load CWN lemmas
    cwn = CwnBase()
    cwn_lemmas = (x["lemma"] for x in cwn.V.values() if x["node_type"] == "lemma")
    cwn_lemmas = list(filter(lambda x: 0 < len(x) <= 2, cwn_lemmas))

    script_dir = Path(__file__).parent
    fasttext_cache = script_dir / '../../resources/fasttext/'
    word_freq_cache = script_dir / '../../resources/as_wordFreq.pickle'
    zhft = Vectors(name='wiki.zh.vec', cache=fasttext_cache)

    with open(word_freq_cache, "rb") as fin:
        wfreq = pickle.load(fin)
    
    cwn_vec = CwnVectors(cwn_lemmas, zhft, wfreq)    
    print("CwnVectors created: ", CwnVectors.cache_path)
