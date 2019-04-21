import logging
from etc import local_config
import pandas as pd
from ..corpus import CorpusFactory
from ..corpus_index import CorpusIndex
from ..enclosing_ngrams import EnclosingNgrams
from ..utils import tqdm
from ..utils import install_data_cache, get_cache_path

logger = logging.getLogger("script_enclosing_ngrams")

def script_enclosing_ngrams(**kwargs):
    DIR_NAME = "enclosing_ngrams"

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

    # load seeds
    seeds_path = get_cache_path(DIR_NAME, f"enclosing_seed.csv")
    with open(seeds_path, "r", encoding="UTF-8") as fin:
        seeds = [x.strip() for x in fin.readlines() if x.strip()]
    if debug:
        seeds = seeds[:10]

    print("loading articles")      
    if debug:
        corpus_articles = take_articles(corpus.articles(), 1000)
        corpus_name += "_dbg"        
    else:        
        corpus_articles = list(tqdm(corpus.articles()))    
    
    corpus_index = CorpusIndex(corpus_name, corpus_articles)
    enc = EnclosingNgrams(corpus_articles, corpus_index)
    enc_results = {}
    for seed_x in tqdm(seeds, ascii=True):
        ng_counter = enc.search_enclosing(seed_x)
        enc_freq = -1
        enc_ngram = ""
        if ng_counter:           
            try:
                # find the frequency of the most common one
                enc_ngram, enc_freq = ng_counter.most_common(1)[0]
            except Exception as ex:
                if not debug:                
                    print(ex)                
                continue        
        enc_results[seed_x] = [enc_ngram, enc_freq]
        if len(enc_results) > 10: break

    # output results    
    install_data_cache(DIR_NAME)
    out_path = get_cache_path(DIR_NAME, f"enclosing_{corpus_name}.csv")
    enc_df = pd.DataFrame.from_dict(enc_results, orient='index', columns=['ngram', 'encfreq'])
    enc_df.to_csv(out_path)
    print("EnclosingNGrams write to ", out_path)

def take_articles(art_it, n=10):
    articles = [next(art_it) for _ in range(n)]
    return articles