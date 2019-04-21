import redis
from functools import partial
from ..corpus import Corpus
from ..utils import tqdm

class RedisCorpus:
    def __init__(self, corpus: Corpus, force_rebuild=False):  
        self.r = redis.StrictRedis(port=6379, db=0)
        self.corpus = corpus        
        if force_rebuild or not self.r.keys(f"{corpus.corpus_name}:*"):
            self.__wrap(corpus)
    
    def __wrap(self, corpus: Corpus):
        print("wrapping corpus with redis")
        _tqdm = partial(tqdm, ascii=True, desc='uploading corpus')        
        for aidx, _, art_x in _tqdm(corpus.articles()):
            rkey =f"{corpus.corpus_name}:{aidx}"            
            self.r.set(rkey, art_x)
    
    def articles(self):
        corpus = self.corpus
        keys = self.r.keys(f"{corpus.corpus_name}:*")
        for k in keys:
            yield (0, 0, self.r.get(k))
    
    def __iter__(self):
        return self.articles()

    def get(self, article_id: int, default=None):
        corpus = self.corpus
        key = f"{corpus.corpus_name}:{article_id}"
        val = self.r.get(key)
        return val        

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise ValueError("Expect an integer as index")
        return self.get(idx)

