import sys
sys.path.append("./")
import pytest
from MWE2019.moe_idioms import MoeIdioms

def test_singleton():
    moe1 = MoeIdioms()
    moe2 = MoeIdioms()
    assert id(moe1.instance) == id(moe2.instance)

def test_data_loaded():
    moe1 = MoeIdioms()
    assert len(moe1.instance.idioms) > 0

def test_magic_method():
    moe1 = MoeIdioms()
    assert len(moe1.instance.idioms) == len(moe1)
    assert moe1["一丘之貉"] != None
    with pytest.raises(KeyError):
        assert moe1["丘之貉"]
    assert "一丘之貉" in moe1
    assert moe1.get("丘之貉", "") == ""
    assert moe1["一丘之貉"]["glossary"]
    assert len(moe1["囫圇吞棗"]["nearsynonym"]) > 1
    assert len(moe1["囫圇吞棗"]["antonym"]) > 1

def test_iteration():
    moe1 = MoeIdioms()
    for idiom, idiom_data in moe1.items():
        assert idiom, idiom_data