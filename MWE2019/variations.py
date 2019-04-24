import re
from typing import Dict, List, Tuple, Iterable
from .corpus import CorpusArticles, Article
from .corpus_index import CorpusIndex
from .variation_db import VariationDb
from .utils import tqdm
from multiprocessing import Pool

NGram = str
Sample = Tuple[int, NGram]
MatchResult = Dict[str, int]
SampleResults = List[Tuple[Sample, MatchResult]]
VarCategory = str
VarPatterns = Tuple[VarCategory, re.Pattern]
VarResult = Tuple[VarCategory, int]
VarResults = Dict[Sample, List[VarResult]]

class VariationFinder:
    def __init__(self, seeds: List[Sample], vardb: VariationDb=None):
        self.samples = seeds
        self.sample_patterns = []                        
        self.vardb = vardb

    def make_variation_patterns(self, seed) -> List[VarPatterns]:
        CJK = "\u3400-\u9fff"
        patterns = []
        seed_chars = list(seed)
        sub_templ = []
        del_templ = []
        for i in range(1, len(seed_chars)-1):
            # substitution
            sub_x = seed_chars.copy()
            sub_x[i] = f"[{CJK}]"
            sub_templ.append("".join(sub_x))            
            patterns.append((f"sub{i+1}", re.compile("".join(sub_x))))

            # deletion
            del_x = seed_chars.copy()
            del del_x[i]
            del_templ.append("".join(del_x))
            patterns.append((f"del{i+1}", re.compile("".join(del_x))))

        # insertions
        ins_templ = []
        for x in seed_chars:
            ins_templ.append(x)
            ins_templ.append(".{,5}?")
        ins_templ = ins_templ[:-1]
        pat_ins = re.compile("".join(ins_templ))
        patterns.append(('ins', pat_ins))
        
        return patterns

    def search_in_corpus(self, corpus_articles: CorpusArticles, 
        corpus_index: CorpusIndex,
        use_cores=1) -> SampleResults:

        sample_results: SampleResults = []        
        for sample_x in tqdm(self.samples, ascii=True, desc="search patterns in corpus"):            
            try:                
                seed_x = sample_x[1]
                hit_indices = corpus_index.search_all_of([seed_x[0], seed_x[-1]])                        
                hit_iter = (corpus_articles[art_i] for art_i in hit_indices)
                match_x = self.search_in_articles(sample_x, hit_iter)
                sample_results.append(match_x)
            except Exception as ex:
                print(ex)
                continue
        return sample_results

    def search_in_articles(self,
            sample_x: Sample,
            articles: Iterable[Article]) -> Tuple[Sample, MatchResult]:
                
        pat_list = self.make_variation_patterns(sample_x[1])        
        match_results = {cat: 0 for cat, _ in pat_list}        
        var_text = {}
        for _, _, art_text in articles:            
            for category, pat in pat_list:
                matches = pat.findall(art_text)
                matches = [x for x in matches if x != sample_x[1]]
                match_results[category] += len(matches)                   
                var_text.setdefault(category, []).extend(matches)        
        
        if self.vardb:
            self.vardb.save(sample_x, var_text)
        # update mat_results to var_results
        sample_results: Tuple[Sample, MatchResult] = (
            sample_x, match_results
        )
        return sample_results

