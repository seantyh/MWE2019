import re
from typing import Dict, List, Tuple
from .corpus import CorpusArticles, Article
from tqdm import tqdm_notebook as tqdm
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
    def __init__(self, seeds: List[Sample]):
        self.samples = seeds
        self.sample_patterns = []
        self.variation_results = {}
        print("pre-compiling variation patterns")
        for sample_x in tqdm(self.samples):
            pat_x = self.make_variation_patterns(sample_x[1])
            self.sample_patterns.append(pat_x)

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
            patterns.append(("sub", re.compile("".join(sub_x))))

            # deletion
            del_x = seed_chars.copy()
            del del_x[i]
            del_templ.append("".join(del_x))
            patterns.append(("del", re.compile("".join(del_x))))

        # sub_pat = re.compile("|".join(sub_templ))
        # del_pat = re.compile("|".join(del_templ))
        # patterns.append(("sub", sub_pat))
        # patterns.append(("del", del_pat))

        # insertions
        ins_templ = []
        for x in seed_chars:
            ins_templ.append(x)
            ins_templ.append(".{,10}?")
        ins_templ = ins_templ[:-1]
        pat_ins = re.compile("".join(ins_templ))
        patterns.append(('ins', pat_ins))
        return patterns

    def search_in_corpus(self, corpus_articles: CorpusArticles, use_cores=1):
        sample_data = list(zip(self.samples, self.sample_patterns))

        from functools import partial

        worker_wrapper = partial(self.search_in_corpus_worker, sample_data)
        with Pool(use_cores) as pool:
            worker_results = list(tqdm(pool.imap(worker_wrapper, corpus_articles, 10), 
                            total=len(corpus_articles),
                            desc="mapping works"))

        for worker_result in tqdm(worker_results, desc="reduce work results"):
            for sample_x, mat_results in worker_result:
                var_result = self.variation_results.setdefault(sample_x, {})
                for category in mat_results.keys():                    
                    var_result[category] = var_result.get(category, 0) + mat_results[category]
        return self.variation_results

    def search_in_corpus_worker(self,
            sample_data: List[Tuple[Sample, List[VarPatterns]]],
            article: Article) -> SampleResults:

        sample_results: SampleResults = []
        _, _, art_text = article
        for sample_x, pat_list in sample_data:
            mat_results = {"ins": 0, "del": 0, "sub": 0}
            for category, pat in pat_list:
                matches = pat.findall(art_text)
                matches = [x for x in matches if x != sample_x[1]]
                mat_results[category] += len(matches)
            sample_results.append((sample_x, mat_results))

        # update mat_results to var_results

        return sample_results

