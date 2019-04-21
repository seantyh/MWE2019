import re
from collections import Counter
from itertools import chain
from typing import Iterable
from ..corpus import CorpusArticles, Article
from ..corpus_index import CorpusIndex

class QieContext:
    def __init__(self, 
            corpus_articles: CorpusArticles, 
            corpus_index: CorpusIndex):
        self.articles = corpus_articles
        self.corpus_index = corpus_index        
    
    def search_context(self, ngram):
        context = []
        try:                            
            hit_indices = self.corpus_index.search_all_of(ngram)            
            hit_iter = (self.articles[art_i] for art_i in hit_indices)            
            mat_context = self.search_sentence(ngram, hit_iter)
            context.extend(mat_context)
        except Exception as ex:
            print(ex)
        return context

    def search_sentence(self, ngram: str, hit_iter: Iterable[Article]) -> Counter:
        ctx_list = []        
        sent_delim = re.compile("[。！？!?\r\n]")
        for _, _, article_text in hit_iter:
            sentences = sent_delim.split(article_text)
            matches = [x for x in sentences if ngram in x]            
            ctx_list.extend(matches)
        return ctx_list
