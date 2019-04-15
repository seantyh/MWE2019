import argparse
from pathlib import Path
from MWE2019.scripts.script_variations import script_variations

parser = argparse.ArgumentParser("MWE2019 command line interface")
parser.add_argument('--task', default='variation', help="task to run")
parser.add_argument('--corpus', help="corpus used in variation finding")
parser.add_argument('--debug', default=False, type=bool, help="use debug mode")
parser.add_argument('--sample-ratio', type=float, help="sample ratio of seeds")
args = parser.parse_args()


if args.task == "variation":
    script_variations(**vars(args))
else:
    print("task not recognized")




