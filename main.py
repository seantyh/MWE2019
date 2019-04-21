import argparse
from pathlib import Path
from MWE2019.scripts import script_variations
from MWE2019.scripts import script_cwnvectors
from MWE2019.scripts import script_cwnmorphgraph
from MWE2019.scripts import script_variations
from MWE2019.scripts import script_build_unigram
from MWE2019.scripts import script_enclosing_ngrams

parser = argparse.ArgumentParser("MWE2019 command line interface")
parser.add_argument('-t', '--task', default='variation', help="task to run",
    choices = ['variations', 'cwnvectors', 'morphgraph', 'unigram',
        "enclosing"])
parser.add_argument('-c', '--corpus', help="corpus used in variation finding")
parser.add_argument('--debug', default=False, type=bool, help="use debug mode")
parser.add_argument('--sample-ratio', default=1, type=float, help="sample ratio of seeds")
args = parser.parse_args()


if args.task == "variations":
    script_variations(**vars(args))
elif args.task == "cwnvectors":
    script_cwnvectors(**vars(args))
elif args.task == "morphgraph":
    script_cwnmorphgraph(**vars(args))
elif args.task == "unigram":
    script_build_unigram(**vars(args))
elif args.task == "enclosing":
    script_enclosing_ngrams(**vars(args))
else:
    print("task not recognized")




