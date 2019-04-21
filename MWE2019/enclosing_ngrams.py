import re
from collections import Counter
from itertools import chain
from typing import Iterable
from .corpus import CorpusArticles, Article
from .corpus_index import CorpusIndex


class EnclosingNgrams:
    def __init__(self, 
            corpus_articles: CorpusArticles, 
            corpus_index: CorpusIndex):
        self.articles = corpus_articles
        self.corpus_index = corpus_index        
    
    def search_enclosing(self, ngram):
        ngram_freq = Counter()
        try:                            
            hit_indices = self.corpus_index.search_all_of(ngram)                        
            hit_iter = (self.articles[art_i] for art_i in hit_indices)            
            match_ngrams = self.search_ngram(ngram, hit_iter)            
            ngram_freq.update(match_ngrams)
        except Exception as ex:
            print(ex)
        return ngram_freq

    def search_ngram(self, ngram: str, hit_iter: Iterable[Article]) -> Counter:
        enc_pat = re.compile(f'[\u3400-\u9fff]{ngram}[\u3400-\u9fff]')
        enc_freq = Counter()
        for _, _, article_text in hit_iter:
            matches = enc_pat.findall(article_text)            
            enc_freq.update(chain.from_iterable([x[:-1], x[1:]] for x in matches))
        
        return enc_freq