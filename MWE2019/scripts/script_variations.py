import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from etc import local_config
from ..utils import tqdm
from ..corpus import CorpusFactory
from ..variations import VariationFinder

logger = logging.getLogger("script_variation")
def script_variations(**kwargs):
    try:
        corpus_name = kwargs["corpus"]        
    except KeyError as ex:
        logger.error("Argument required: %s", ex)
        return
    rnd_seed = kwargs.get('seed', 21563)

    if corpus_name == "apple":
        corpus = CorpusFactory.GetNewsCorpus(local_config.CORPUS_DIR + "/Apple_text")
    elif corpus_name == "chinatimes":
        corpus = CorpusFactory.GetNewsCorpus(local_config.CORPUS_DIR + "/China_text")
    elif corpus_name == "ptt":
        corpus = CorpusFactory.GetPttCorpus(local_config.CORPUS_DIR + "/PTT")
    else:
        logger.error("unrecognized corpus")
        return
    
    NGRAM4_DFRAME_PATH = Path(__file__) / "../../data/ngram4_variation.csv"
    if NGRAM4_DFRAME_PATH.exists():
        ngram4 = pd.read_csv(NGRAM4_DFRAME_PATH)
    else:
        ngram4 = build_ngram4_dframe()
    
    seeds = sample_seeds(rnd_seed, ngram4)
    corpus_articles = sample_articles(corpus.articles())

    var_finder = VariationFinder(seeds)    
    result_buf = var_finder.search_in_corpus(corpus_articles, 4)
    
    update_results(var_finder.variation_results)

def build_ngram4_dframe():
    NGRAM_BASE_PATH = Path(__file__) / "../../resources/ngram_4.csv"
    ngram4 = pd.read_csv(NGRAM_BASE_PATH)
    ngram4["insertion"] = np.nan
    ngram4["substitution"] = np.nan
    ngram4["deletion"] = np.nan
    return ngram4

def sample_seeds(rnd_seed, ngram4):
    SAMPLE_RATE = 0.001
    np.random.seed(rnd_seed)
    rand_vec = np.random.random(ngram4.shape[0])
    samples = []
    for rnd, (ridx, row) in tqdm(zip(rand_vec, ngram4.iterrows()), total=len(rand_vec)):
        if rnd < SAMPLE_RATE:
            if not row.isnull().any(): continue
            samples.append((ridx, row.ngram))
    return samples

def sample_articles(art_it, ratio=0.05):
    articles = [x for x in art_it if np.random.random() < ratio] 
    return articles

def update_results(var_results: Dict[str, int], ngram4: pd.DataFrame):
    for sample_x, var_result in tqdm(var_results.variation_results.items()): 
        sample_id = sample_x[0]    
        sub = var_result["sub"]
        ins = var_result["ins"]
        delt = var_result["del"]
        ngram4.loc[sample_id, "substitution"] = sub
        ngram4.loc[sample_id, "insertion"] = ins
        ngram4.loc[sample_id, "deletion"] = delt
    