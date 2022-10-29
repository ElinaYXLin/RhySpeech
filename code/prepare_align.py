import argparse

import yaml

from preprocessor import ljspeech, aishell3, libritts


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config)
    if "AISHELL3" in config["dataset"]:
        aishell3.prepare_align(config)
    if "LibriTTS" in config["dataset"]:
        libritts.prepare_align(config)


if __name__ == "__main__":
 #   config = yaml.load(open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    config = yaml.load(open("./config/LibriTTS/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    main(config)
