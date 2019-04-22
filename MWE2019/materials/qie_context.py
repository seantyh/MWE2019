import re
from collections import defaultdict
from typing import Iterable
import numpy as np
from ..utils import tqdm
from ..corpus import CorpusArticles, Article
from ..corpus_index import CorpusIndex

class QieContext:
    def __init__(self,
            corpus_articles: CorpusArticles):
        self.articles = corpus_articles

    def search_context(self, ngrams):
        context = defaultdict(list)
        sent_delim = re.compile("[。！？!?\r\n]")        
        for _, _, article_text in tqdm(self.articles):
            sentences = sent_delim.split(article_text)
            for ngram_x in ngrams:
                matches = [sent_x for sent_x in sentences if ngram_x in sent_x]
                context[ngram_x].extend(matches)            
        return context
