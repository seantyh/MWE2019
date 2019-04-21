import sys
sys.path.append("./")
from MWE2019.enclosing_ngrams import EnclosingNgrams
from MWE2019.corpus_index import CorpusIndex

def test_enclosing():
    corpus = [[0,0,"這是一篇測試文章中"], [1,1,"這是篇測試文章上"], [2,2,"這是一篇測試文章裡"]]
    index = CorpusIndex('test_enclosing', corpus)
    enc = EnclosingNgrams(corpus, index)
    mat = enc.search_enclosing("測試文章")    
    assert mat.most_common(1)[0][1] == 3
    