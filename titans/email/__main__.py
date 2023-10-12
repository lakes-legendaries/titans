from argparse import ArgumentParser

import yaml

from titans.email.send import SendEmails


# read command line
parser = ArgumentParser(description="Mass-send emails using MS Graph API")
parser.add_argument(
    "yamlconfig",
    help="Configuration YAML filename",
)
yaml_fname = parser.parse_args().yamlconfig

# read yaml file
config = yaml.safe_load(open(yaml_fname, "r"))

# send emails
SendEmails(**config)
