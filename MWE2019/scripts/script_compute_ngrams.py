import logging
import pickle
from itertools import islice
from collections import defaultdict
import pandas as pd
from etc import local_config
from ..corpus import CorpusFactory
from ..utils import tqdm
from ..utils import install_data_cache, get_cache_path

logger = logging.getLogger("script_ngrams")

def script_compute_ngrams(**kwargs):
    DIR_NAME = "cache_ngrams"

    try:
        corpus_name = kwargs["corpus"]
    except KeyError as ex:
        logger.error("Argument required: %s", ex)
        return
    
    debug = kwargs.get('debug', False)    
    
    if corpus_name == "apple":
        corpus = CorpusFactory.GetNewsCorpus(local_config.CORPUS_DIR + "/Apple_text")
    elif corpus_name == "chinatimes":
        corpus = CorpusFactory.GetNewsCorpus(local_config.CORPUS_DIR + "/China_text")
    elif corpus_name == "ptt":
        corpus = CorpusFactory.GetPttCorpus(local_config.CORPUS_DIR + "/PTT")
    else:
        logger.error("unrecognized corpus")
        return

    seeds_path = get_cache_path("enclosing_ngrams", f"enclosing_seed.csv")
    with open(seeds_path, "r", encoding="UTF-8") as fin:
        seeds = [x.strip() for x in fin.readlines() if x.strip()]
    if debug:
        seeds = seeds[:10]

    ngram_counts = defaultdict(lambda: 0)
    for art_i, _, art_x in tqdm(corpus.articles(), ascii=True):
        for seed_x in seeds:
            ngram_counts[seed_x] += art_x.count(seed_x)

    # output results    
    install_data_cache(DIR_NAME)
    out_path = get_cache_path(DIR_NAME, f"ngram_seeds_{corpus_name}.csv")
    ngram_df = pd.DataFrame.from_dict(ngram_counts, orient='index', columns=['exact_freq'])
    ngram_df.to_csv(out_path)
    print("NGram exact frequency write to ", out_path)


    