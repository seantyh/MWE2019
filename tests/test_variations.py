import sys
sys.path.append("./")
from MWE2019.variations import VariationFinder
from MWE2019.corpus_index import CorpusIndex

def test_variations():
    articles = [[0, "a", "中文測試篇章"], [1, "b", "英文測試一篇文章"], [2, "c", "還是測文章也可以"]]
    var_finder = VariationFinder([(0, "測試文章")], None)
    corpus_index = CorpusIndex("test_variation", articles) 
    mats = var_finder.search_in_corpus(articles, corpus_index)
    print(mats)
    assert mats[0][1]["sub3"] == 1
    assert mats[0][1]["del2"] == 1
    assert mats[0][1]["ins"] == 1
    