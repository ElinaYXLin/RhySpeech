import argparse

import yaml

from preprocessor.preprocessor import Preprocessor

# first run the line below:
# ./montreal-forced-aligner/bin/mfa_align raw_data/LibriTTS/ lexicon/librispeech-lexicon.txt english preprocessed_data/LibriTTS


print("run")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to preprocess.yaml", required=False, 
    default="config/LJSpeech/preprocess.yaml")

    # use additional file to bring in parameters, need to use "--config path-to-file" when running
    args = parser.parse_args()
    # args = parser.parse_known_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    # preprocessor is used for the actual preprocessing (two lines below are the most important of the file)
    preprocessor = Preprocessor(config)
 #   preprocessor.build_from_path()
    preprocessor.rhy_conc_build_from_path()
