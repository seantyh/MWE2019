import pickle
from typing import List, Tuple, Set, Dict, Iterable
from pathlib import Path
from .corpus import CorpusArticles
from .utils import tqdm

Character = str
CorpusCacheId = str
ArticleIndex = int
ArticlesIndexData = Dict[Character, List[ArticleIndex]]
CorpusIndexCache = Tuple[CorpusCacheId, ArticlesIndexData]

class CorpusIndex:
    def __init__(self, corpus_name: str, articles:CorpusArticles):
        self.corpus_name = corpus_name
        self.cache: CorpusIndexCache = {}
        self.cache_path = Path(__file__).parent / \
            f"../data/corpus_cache/{self.corpus_name}.index.pkl"
        try:
            print("try loading index...", end='')
            self.load_index_cache()
            print('OK')
        except FileNotFoundError as ex:            
            self.build_index(articles)
            print("Saving index...", end='')
            self.save_index_cache()
            print("Done")
    
    def load_index_cache(self):        
        with open(self.cache_path, "rb") as fin:
            self.cache = pickle.load(fin)                
    
    def save_index_cache(self):        
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as fout:
            pickle.dump(self.cache, fout) 

    def build_index(self, articles: CorpusArticles):
        cache = self.cache
        art_counter = 0
        for fi, f, text in tqdm(articles, ascii=True, desc="building article index"):
            for ch in set(text):
                if not "\u3400" < ch < "\u9fff":
                    continue                
                cache.setdefault(ch, []).append(art_counter)
            art_counter += 1

    def search_all_of(self, chars:Iterable[str])\
        -> CorpusArticles:
        cache = self.cache
        if chars:
            hit_articles = set(cache.get(chars[0], []))
        for ch in chars[1:]:            
            hit_articles.intersection_update(cache.get(ch, []))
            
        return list(hit_articles)
        