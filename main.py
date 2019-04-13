import argparse
from pathlib import Path
from MWE2019.scripts.script_variations import script_variations

parser = argparse.ArgumentParser("MWE2019 command line interface")
parser.add_argument('--task', default='variation')
args = parser.parse_args()


if args.task == "variation":
    script_variations(**vars(args))
else:
    print("task not recognized")




