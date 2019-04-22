from ..utils import install_data_cache, get_cache_path
import pandas as pd
from functools import reduce

def compute_enclosing(base_df):        

    corpus_names = ["apple", "chinatimes", "ptt"]

    def read_enc_data(name):
        enc_path = get_cache_path('enclosing_ngrams', f"enclosing_{name}.csv")
        data = pd.read_csv(enc_path, index_col=0) \
                    .rename(columns=lambda x: x+"."+name[0].upper())    
        return data

    enc_iter = (read_enc_data(x) for x in corpus_names)
    enc_data = reduce(lambda x, y: x.join(y), enc_iter)
    encfreq_colmask = enc_data.columns.str.startswith("encfreq")
    enc_freq_data = pd.DataFrame({
        "enc_freq": enc_data.loc[:, encfreq_colmask].sum(1)
    }, index=enc_data.index)
    return base_df.join(enc_freq_data, how='inner') 