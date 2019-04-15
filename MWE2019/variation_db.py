from pymongo import MongoClient, ASCENDING
from typing import List, Dict, Any

class VariationDb:
    def __init__(self, mongo_inst, coll_name):                
        if mongo_inst:
            self.db = mongo_inst.mwe
            if coll_name not in self.db.collection_names():
                coll = self.db[coll_name]
                coll.create_index([('ngram_id', ASCENDING)])
            self.coll = self.db[coll_name]

        else:
            self.db = None
            self.coll = None
    
    def save(self, sample: "Sample", data: Dict[str, Any]):
        if not self.coll:            
            print("WARNING: no valid mongo instance")
            return
        var_doc = {
            "ngram_id": sample[0], 
            "ngram": sample[1],             
            "variations": data}
        ins_res = self.coll.insert_one(var_doc)           
        return ins_res.inserted_id
    
    def load(self, sample: "Sample"):
        pass