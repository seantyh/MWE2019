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
    def __init__(self, seeds: List[Sample], vardb: VariationDb):
        self.samples = seeds
        self.sample_patterns = []                
        for sample_x in tqdm(self.samples, ascii=True, desc="precompile patterns"):
            pat_x = self.make_variation_patterns(sample_x[1])
            self.sample_patterns.append(pat_x)
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
            ins_templ.append(".{,10}?")
        ins_templ = ins_templ[:-1]
        pat_ins = re.compile("".join(ins_templ))
        patterns.append(('ins', pat_ins))
        return patterns

    def search_in_corpus(self, corpus_articles: CorpusArticles, 
        corpus_index: CorpusIndex,
        use_cores=1) -> SampleResults:

        sample_results: SampleResults = []
        sample_data = list(zip(self.samples, self.sample_patterns))
        for sample_data_x in tqdm(sample_data, ascii=True, desc="search patterns in corpus"):
            sample_x, _ = sample_data_x
            seed_x = sample_x[1]
            hit_indices = corpus_index.search_all_of([seed_x[0], seed_x[-1]])                        
            hit_iter = (corpus_articles[art_i] for art_i in hit_indices)
            match_x = self.search_in_articles(sample_data_x, hit_iter)
            sample_results.append(match_x)
        return sample_results

    def search_in_articles(self,
            sample_data: Tuple[Sample, List[VarPatterns]],
            articles: Iterable[Article]) -> Tuple[Sample, MatchResult]:

        match_results = {cat: 0 for cat, _ in sample_data[1]}
        sample_x, pat_list = sample_data
        var_text = {}
        for _, _, art_text in articles:            
            for category, pat in pat_list:
                matches = pat.findall(art_text)
                matches = [x for x in matches if x != sample_x[1]]
                match_results[category] += len(matches)           
                var_text[category] = matches

        self.vardb.save(sample_x, var_text)
        # update mat_results to var_results
        sample_results: Tuple[Sample, MatchResult] = (
            sample_x, match_results
        )
        return sample_results

