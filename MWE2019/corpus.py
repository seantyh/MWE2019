import os
import pdb
import logging
import re
from typing import Iterable, Tuple

Filename = str
Text = str
CorpusArticles = Iterable[Tuple[int, Filename, Text]]
logger = logging.getLogger("tetra")

class CorpusFactory:
    @classmethod
    def GetPttCorpus(self, path):        
        return PTTCorpus(path)
    
    @classmethod
    def GetNewsCorpus(self, path):
        return NewsCorpus(path)

class Corpus:
    alpha_re = re.compile("[\w]+")    
    def __init__(self, corpus_path):        
        if not os.path.exists(corpus_path):
            pdb.set_trace()
            raise ValueError("Cannot find corpus: %s" % (corpus_path,))
        
        self.corpus_name = os.path.basename(corpus_path)
        self.base_path = corpus_path

        flist = os.listdir(self.base_path)
        self.file_count = len(flist)

class PTTCorpus(Corpus):
    DAYS_OF_WEEK = ("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
    def __init__(self, corpus_name):
        super().__init__(corpus_name)
        self.skip_counter = 0
        self.article_counter = 0
        self.file_count = 0
        
    def articles(self):
        flist = os.listdir(self.base_path)        
        for subf in flist:           
            subcorpus_path = os.path.join(self.base_path, subf) 
            if not os.path.isdir(subcorpus_path): continue
            for fi, f, art in self.get_subcorpus(subcorpus_path):
                yield ((fi, f, art))
            
    def get_subcorpus(self, subcorpus_path):
        for fi, f in enumerate(os.listdir(subcorpus_path)):            
            if not f.endswith(".txt"): continue
            fpath = os.path.join(subcorpus_path, f)
            fin = open(fpath, "r", encoding="UTF-8", 
                    errors = "backslashreplace", newline = "\r\n")
            for art in self.get_subcorpus_articles(fin):
                yield((fi, f, art))
            logger.info("[%s] %s:%d / %d" % (self.corpus_name, f, fi,
                self.file_count))
            self.file_count += 1
            fin.close()                
            
    def get_subcorpus_articles(self, fs):
        ln_mode = 0
        title = ""
        url = ""
        prev_line = ""
        for ln in fs.readlines():
            lp_idx = ln.find("(")
            # pdb.set_trace()
            if ln.startswith("https://"): 
                ln_mode = 1
            elif ln_mode == 1 and lp_idx > 0 and ln.find(")") > lp_idx:
                ln_mode = 2
            elif ln_mode == 2 and Corpus.alpha_re.match(ln) != None:
                ln_mode = 3
            elif ln_mode == 3:
                ln_mode = 4
            elif ln_mode == 4 and ln[0:3] in PTTCorpus.DAYS_OF_WEEK:
                ln_mode = 5
            elif ln_mode == 5 and len(ln.strip()) > 0:
                ln_mode = 6
            elif ln_mode == 6 and len(ln.strip()) == 0:
                ln_mode = 0
            else:
                ln_mode = -1
                        
            if ln_mode == 1:
                url = ln.strip()
            elif ln_mode == 4:
                title = ln.strip()
            elif ln_mode == 0:
                self.article_counter += 1
                yield prev_line
            
            prev_line = ln

class NewsCorpus(Corpus):
    date_re = re.compile("[\d\/]\s+$")
    def __init__(self, corpus_name):
        super().__init__(corpus_name)
        self.skip_counter = 0
        self.article_counter = 0


    def articles(self):
        flist = os.listdir(self.base_path)
        for fi, f in enumerate(flist):
            if not f.endswith(".txt"): continue
            fpath = os.path.join(self.base_path, f)
            fin = open(fpath, "r", encoding="UTF-8", 
                    errors = "backslashreplace", newline = "\r\n")
            for art in self.get_articles(fin):
                yield((fi, f, art))
            logger.info("[%s] %s:%d / %d" % (self.corpus_name, f, fi,
                self.file_count))
            fin.close()

    def get_articles(self, fs):
        ln_mode = 0
        date = ""
        prev_ln = ""
        title_line = ""
        try:
            for ln in fs.readlines():
                if len(ln.strip()) == 0:
                    ln_mode = 0
                else:
                    ln_mode += 1
                
                if ln_mode == 0:
                    self.article_counter += 1                    
                    yield title_line + prev_ln
                elif ln_mode == 1:
                    title_line = ln
                else:
                    pass
                
                prev_ln = ln
        except (UnicodeDecodeError, TypeError) as ex:
            print(ex)
            pdb.set_trace()

