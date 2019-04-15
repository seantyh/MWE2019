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


    print("loading articles")
    if DEBUG:
        corpus_articles = take_articles(corpus.articles(), 1000)
        NGRAM4_DFRAME_PATH = Path(__file__).parent / "../../data/ngram4_variation_dbg.csv"
        corpus_name += "_dbg"
    else:
        NGRAM4_DFRAME_PATH = Path(__file__).parent / "../../data/ngram4_variation.csv"
        corpus_articles = list(tqdm(corpus.articles()))

    
    ## connect to mongodb
    mongo = MongoClient(local_config.MONGO_HOST, local_config.MONGO_PORT)
    vardb = VariationDb(mongo, corpus_name)

    ## extract ngrams
    if NGRAM4_DFRAME_PATH.exists():
        ngram4 = pd.read_csv(NGRAM4_DFRAME_PATH)
    else:
        ngram4 = build_ngram4_dframe(DEBUG)
    seeds = sample_seeds(rnd_seed, ngram4, sample_ratio)

    ## initialize VariationFinder
    var_finder = VariationFinder(seeds, vardb)
    corpus_index = CorpusIndex(corpus_name, corpus_articles)
    sample_results = var_finder.search_in_corpus(corpus_articles, corpus_index, 4)

    ## prepare writing out
    write_snapshot(ngram4, str(NGRAM4_DFRAME_PATH))
    update_results(sample_results, ngram4)
    print('Saving ngram4_variation to file...', end='')
    ngram4.to_csv(NGRAM4_DFRAME_PATH, index=False)
    print('Done')

def build_ngram4_dframe(is_debug=False):
    NGRAM_BASE_PATH = Path(__file__).parent / "../../resources/ngram_4.csv"
    ngram4 = pd.read_csv(NGRAM_BASE_PATH)

    if is_debug:
        ngram4 = ngram4.iloc[:1000, :]
    else:
        ## see etc/select_ngrams.ipynb for the rationale of 322 threshold
        ngram4 = ngram4.loc[ngram4.freq >= 322, :]
    for field_names in FIELD_NAMES:
        ngram4[field_names] = -1
    return ngram4

def sample_seeds(rnd_seed, ngram4, sample_ratio=0.05):
    np.random.seed(rnd_seed)
    ngram4_todo = ngram4.loc[ngram4.ins >= 0, :]    
    rand_vec = np.random.random(ngram4_todo.shape[0])    
    selected_idx = np.argwhere(rand_vec < sample_ratio).flatten().tolist()

    samples = []
    for ridx, row in ngram4_todo.iloc[selected_idx, :].iterrows():        
        samples.append((ridx, row.ngram))
    return samples

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
