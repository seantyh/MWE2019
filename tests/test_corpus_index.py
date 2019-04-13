import sys
sys.path.append("./")
from MWE2019.corpus_index import CorpusIndex

def test_corpus_index():
    articles = [[0, "a", "中文測試文章"], [1, "b", "英文測試文章"], [2, "c", "還是這樣子有文章"]]
    cindex = CorpusIndex("test_corpus", articles)
    hits = cindex.search_all_of("測試")
    assert sorted(hits) == [0, 1]
    hits = cindex.search_all_of("文章")
    assert sorted(hits) == [0, 1, 2]
    hits = cindex.search_all_of("這樣子")
    assert sorted(hits) == [2]