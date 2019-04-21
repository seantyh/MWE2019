from collections import Counter
from pathlib import Path
import pickle
from ..utils import install_data_cache, get_cache_path
from ..utils import tqdm
import numpy as np
import pandas as pd

class NGram4List:
    def __init__(self):
        try:
            self.load_cache()
        except:
            print("Building ngrams list...", end='')
            self.build_ngrams_list()
            print("Done")
            print("Computing pmi...", end='')
            self.compute_pmi()
            print("Done")
            self.write_cache()            

    def __contains__(self, x):
        return x in self.merged_df.index

    def __iter__(self):
        for ngram in self.merged_df.index:
            yield ngram
    
    def get_data(self, ngram):
        return self.merged_df.loc[ngram, :]

    def build_ngrams_list(self):
        var_df = []
        for corpus_name in ('apple', 'chinatimes', 'ptt'):
            var_path = Path(__file__).parent / f"../../data/variations/qievars_{corpus_name}.csv"
            if not var_path.exists():
                continue
            corpus_df = pd.read_csv(var_path)
            corpus_df.set_index(['ngram', 'freq'], inplace=True)
            corpus_df.rename(columns={x: x+f'.{corpus_name[0].upper()}' 
                                for x in corpus_df.columns}, inplace=True)            
            var_df.append(corpus_df)
            
        merged_df = var_df[0]
        for df_x in var_df[1:]:
            merged_df = merged_df.join(df_x)
        merged_df.reset_index("freq", inplace=True)
        self.merged_df = merged_df
        return merged_df

    def compute_pmi(self):
        # load unigram        
        unigram_path = get_cache_path("cache_corpus_unigram", "unigram.pkl")
        with open(unigram_path, "rb") as fin:
            unigram = pickle.load(fin)    
            chfreq = Counter({k: v for k, v in unigram.items() if "\u3400" < k < "\u9fff"})
        
        merged_df = self.merged_df
        merged_df.pmi = np.nan        
        for ridx, row in tqdm(merged_df.iterrows(), 
                ascii=True, total=merged_df.shape[0]):
            ngram = ridx
            cf_vec = [chfreq.get(x, 1) for x in ngram]
            pmi = np.log(row.freq) - np.sum(np.log(cf_vec))
            merged_df.loc[ridx, "pmi"] = pmi
        self.merged_df = merged_df

    def load_cache(self):
        cache_path = get_cache_path('cache_ngrams_list', 'ngrams_var_list.csv')
        self.merged_df = pd.read_csv(cache_path)
        self.merged_df.set_index('ngram', inplace=True)
        print("NGram4List loaded: ", cache_path)
    
    def write_cache(self):
        install_data_cache('cache_ngrams_list')
        cache_path = get_cache_path('cache_ngrams_list', 'ngrams_var_list.csv')
        self.merged_df.to_csv(cache_path)
        print("NGram4List cache to ", cache_path)