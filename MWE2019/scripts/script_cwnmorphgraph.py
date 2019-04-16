import pickle
import gzip
import json
from CwnGraph import CwnBase
from pathlib import Path
from ..cwn_vectors import CwnVectors
from ..cwn_morph_graph import CwnMorphGraph

def script_cwnmorphgraph(**kwargs):
    cwn = CwnBase()
    debug = kwargs.get("debug", False)
    cwn_vectors = CwnVectors()
    cwn_mgraph = CwnMorphGraph(cwn_vectors, cwn, debug=debug)
    print("Morphology Graph: ")
    print("|V|: ", cwn_mgraph.G.number_of_nodes())
    print("|E|: ", cwn_mgraph.G.number_of_edges())
    cwn_mgraph.to_dot()