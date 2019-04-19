import csv
from pathlib import Path

class MoeIdioms:
    instance = None
    def __init__(self):
        inst = MoeIdioms.instance
        if not inst:
            print("initialize new instance")
            inst = _MoeIdioms()        
        MoeIdioms.instance = inst

    def __len__(self):
        return len(MoeIdioms.instance.idioms)
    
    def __contains__(self, item):
        return item in MoeIdioms.instance.idioms

    def __iter__(self):
        return iter(MoeIdioms.instance.idioms)
    
    def __getitem__(self, key):
        return MoeIdioms.instance.idioms[key]

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name, val):
        return setattr(self.instance, name, val)
    

class _MoeIdioms:    
    def __init__(self):
        data_path = Path(__file__).parent / \
            "../resources/dict_idioms_2011_20190329.csv"
        self.idioms = {}
        self.load_idioms(data_path)        
    
    def get(self, name, default):
        return MoeIdioms.instance.idioms.get(name, default)
    
    def items(self):
        return MoeIdioms.instance.idioms.items()

    def load_idioms(self, data_path):
        fin = open(data_path, "r", encoding="UTF-8")        
        csvreader = csv.reader(fin)
        next(csvreader)
        idiom_data = {}
        for row in csvreader:
            idiom = row[1]
            glossary = row[4]
            source = row[5]
            nearsyn = row[9].split("\n")
            antonym = row[10].split("\n")
            references = row[11]
            idiom_data[idiom] = {
                "glossary": glossary,
                "source": source,
                "nearsynonym": nearsyn,
                "antonym": antonym,
                "references": references
            }

        self.idioms = idiom_data

