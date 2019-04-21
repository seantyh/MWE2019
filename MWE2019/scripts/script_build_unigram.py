import sys
sys.path.append("../")
from etc import local_config
import pickle
from MWE2019.materials import NGram4List, MoeIdioms
from MWE2019.corpus import CorpusFactory
from MWE2019.utils import tqdm
from MWE2019.utils import install_data_cache, get_cache_path
from collections import Counter

def script_build_unigram(**kwargs):
    debug = kwargs.get("debug", False)
    ptt = CorpusFactory.GetPttCorpus(local_config.CORPUS_DIR + "/PTT")
    ctimes = CorpusFactory.GetNewsCorpus(local_config.CORPUS_DIR + "/China_text")
    apple = CorpusFactory.GetNewsCorpus(local_config.CORPUS_DIR + "/Apple_text")
    char_freq = Counter()
    print("iterating over ptt")
    for i, _, article_x in tqdm(ptt.articles()):
        char_freq.update(article_x)    
        if debug: break
    print("iterating over apple")
    for i, _, article_x in tqdm(apple.articles()):
        char_freq.update(article_x)
        if debug: break
    print("iterating over chimatimes")
    for i, _, article_x in tqdm(ctimes.articles()):
        char_freq.update(article_x)
        if debug: break
    
    install_data_cache("cache_corpus_unigram")
    cache_path = get_cache_path("cache_corpus_unigram", "unigram.pkl")
    with open(cache_path, "wb") as fout:
        pickle.dump(char_freq, fout)
    
    print("Unigram cache write to: ", cache_path)
    
    
    

