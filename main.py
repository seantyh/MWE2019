import argparse
from pathlib import Path

parser = argparse.ArgumentParser("MWE2019 command line interface")
parser.add_argument('-t', '--task', default='variation', help="task to run",
    choices = ['variations', 'cwnvectors', 'morphgraph', 'unigram',
        "enclosing", "ngrams"])
parser.add_argument('-c', '--corpus', help="corpus used in variation finding")
parser.add_argument('--debug', default=False, type=bool, help="use debug mode")
parser.add_argument('--sample-ratio', default=1, type=float, help="sample ratio of seeds")
args = parser.parse_args()


if args.task == "variations":   
    from MWE2019.scripts.script_variations import script_variations
    script_variations(**vars(args))

elif args.task == "cwnvectors":
    from MWE2019.scripts.script_cwnvectors import script_cwnvectors
    script_cwnvectors(**vars(args))

elif args.task == "morphgraph":
    from MWE2019.scripts.script_cwnmorphgraph import script_cwnmorphgraph
    script_cwnmorphgraph(**vars(args))

elif args.task == "unigram":
    from MWE2019.scripts.script_build_unigram import script_build_unigram
    script_build_unigram(**vars(args))

elif args.task == "enclosing":
    from MWE2019.scripts.script_enclosing_ngrams import script_enclosing_ngrams
    script_enclosing_ngrams(**vars(args))

elif args.task == "ngrams":
    from MWE2019.scripts.script_compute_ngrams import script_compute_ngrams
    script_compute_ngrams(**vars(args))
    
else:
    print("task not recognized")




