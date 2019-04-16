import argparse
from pathlib import Path
from MWE2019.scripts import script_variations
from MWE2019.scripts import script_cwnvectors

parser = argparse.ArgumentParser("MWE2019 command line interface")
parser.add_argument('-t', '--task', default='variation', help="task to run",
    choices = ['variations', 'cwnvectors'])
parser.add_argument('-c', '--corpus', help="corpus used in variation finding")
parser.add_argument('--debug', default=False, type=bool, help="use debug mode")
parser.add_argument('--sample-ratio', default=1, type=float, help="sample ratio of seeds")
args = parser.parse_args()


if args.task == "variations":
    script_variations(**vars(args))
elif args.task == "cwnvectors":
    script_cwnvectors(**vars(args))
else:
    print("task not recognized")




