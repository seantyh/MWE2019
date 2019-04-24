import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from etc import local_config
from datetime import datetime
from pymongo import MongoClient
from ..utils import tqdm
from ..corpus import CorpusFactory
from ..corpus_index import CorpusIndex
from ..variations import VariationFinder, SampleResults
from ..variation_db import VariationDb
from ..utils import get_cache_path, install_data_cache

logger = logging.getLogger("script_variation")
FIELD_NAMES = ["sub2", "del2", "sub3", "del3", "ins"]

def script_variations(**kwargs):    

    try:
        corpus_name = kwargs["corpus"]
    except KeyError as ex:
        logger.error("Argument required: %s", ex)
        return
    rnd_seed = kwargs.get('seed', 21563)
    DEBUG = kwargs.get('debug', False)
    sample_ratio = kwargs.get("sample_ratio", 0.05)

    if corpus_name == "apple":
        corpus = CorpusFactory.GetNewsCorpus(local_config.CORPUS_DIR + "/Apple_text")
    elif corpus_name == "chinatimes":
        corpus = CorpusFactory.GetNewsCorpus(local_config.CORPUS_DIR + "/China_text")
    elif corpus_name == "ptt":
        corpus = CorpusFactory.GetPttCorpus(local_config.CORPUS_DIR + "/PTT")
    else:
        logger.error("unrecognized corpus")
        return

    install_data_cache('variations')

    print("loading articles")
    if DEBUG:
        corpus_articles = take_articles(corpus.articles(), 1000)
        corpus_name += "_dbg"            
    else:
        rs = np.random.RandomState(5222)  # pylint: disable=no-member
        corpus_articles = [x for x in tqdm(corpus.articles()) if rs.rand() < sample_ratio]
        corpus_name += "_sample"
    NGRAM4_DFRAME_PATH = get_cache_path('variations', f"ngrams_vars_{corpus_name}.csv")

    ## connect to mongodb
    mongo = MongoClient(local_config.MONGO_HOST, local_config.MONGO_PORT)
    vardb = VariationDb(mongo, corpus_name)

    ## extract ngrams
    if NGRAM4_DFRAME_PATH.exists():
        ngram4 = pd.read_csv(NGRAM4_DFRAME_PATH)
    else:
        ngram4 = build_ngram4_dframe(DEBUG)
    seeds = list(ngram4.ngram.iteritems())
    
    ## initialize VariationFinder
    var_finder = VariationFinder(seeds, vardb)
    corpus_index = CorpusIndex(corpus_name, corpus_articles)
    sample_results = var_finder.search_in_corpus(corpus_articles, corpus_index, 4)

    ## prepare writing out
    write_snapshot(ngram4, str(NGRAM4_DFRAME_PATH))
    update_results(sample_results, ngram4)
    print('Saving qie variation to file...', end='')
    ngram4.to_csv(NGRAM4_DFRAME_PATH, index=False)
    print('Done')

def build_ngram4_dframe(is_debug=False):
    NGRAM_BASE_PATH = get_cache_path('cache_ngrams_list', 'ngrams_list.csv')
    ngram4 = pd.read_csv(NGRAM_BASE_PATH)
    
    for field_names in FIELD_NAMES:
        ngram4[field_names] = -1    
    return ngram4

def sample_articles(art_it, ratio=0.05):
    articles = [x for x in art_it if np.random.random() < ratio]
    return articles

def take_articles(art_it, n=10):
    articles = [next(art_it) for _ in range(n)]
    return articles

def update_results(sample_results: SampleResults, ngram4: pd.DataFrame):
    for sample_x, mat_result in tqdm(sample_results, ascii=True, desc="update results"):
        sample_id = sample_x[0]        
        for field_names in FIELD_NAMES:                        
            ngram4.loc[sample_id, field_names] = mat_result[field_names]


def write_snapshot(ngram_df, dframe_path):
    print("taking snapshot...", end='')
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    ngram_df.to_csv(dframe_path.replace(".csv", f"_{timestamp}.csv"), index=False)
    print("Done")
