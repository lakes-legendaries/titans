from argparse import ArgumentParser

import yaml

from titans.pdfset.writer import SetWriter


# parse args
parser = ArgumentParser("Compile set PDFs from card PNGs")
parser.add_argument(
    "-s",
    "--set",
    default="titans/pdfset/genesis.yaml",
    help="Yaml file. Unpacked to construct SetWriter",
)
args = parser.parse_args()

# read yaml
config = yaml.safe_load(open(args.set, "r"))

# create set
SetWriter(**config)
