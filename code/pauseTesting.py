import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence
import nltk

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = phones.replace("\{[^\w\s]?\}", "") #new
    phones = phones.replace(",", "") #new
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, batchs):
# # rhyspeech
#     tp = 0 # true positive
#     fp = 0
#     tn = 0
#     fn = 0 # false negative
#     cnt = 0
#     for batch in batchs:
#         batch = to_device(batch, device)
#         with torch.no_grad():
#             # Forward
#             output = model(
#                 *(batch[2:])
#             )
#             rhyPred = output[10]
#             pauseTruth = []
#             for ele in batch[0]:
#                 pausePath = "preprocessed_data/LJSpeech/pause/LJSpeech-pause-"+ele+".npy"
#                 pauseTruth.append(np.load(pausePath).tolist())
#             for i in range(len(pauseTruth)):
#                 for j in range(len(pauseTruth[i])):
#                     cnt += 1
#                     if (pauseTruth[i][j]<0.5 and rhyPred[i][j]<0.5):
#                         tn += 1
#                     if (pauseTruth[i][j]<0.5 and rhyPred[i][j]>=0.5):
#                         fp += 1
#                     if (pauseTruth[i][j]>=0.5 and rhyPred[i][j]<0.5):
#                         fn += 1
#                     if (pauseTruth[i][j]>=0.5 and rhyPred[i][j]>=0.5):
#                         tp += 1
#     print(f"Total: {cnt}")
#     print(f"True Positve: {tp}")
#     print(f"False Positve: {fp}")
#     print(f"True Negative: {tn}")
#     print(f"False Negative: {fn}")

    # baseline
    tp = 0 # true positive
    fp = 0
    tn = 0
    fn = 0 # false negative
    cnt = 0
    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:])
            )
            rhyPred = output[10]
            pauseTruth = []
            for ele in batch[0]:
                pausePath = "preprocessed_data/LJSpeech/pause/LJSpeech-pause-"+ele+".npy"
                pauseTruth.append(np.load(pausePath).tolist())
            for i in range(len(pauseTruth)):
                for j in range(len(pauseTruth[i])):
                    # if (phoneme)
                    cnt += 1
                    if (pauseTruth[i][j]<0.5 and rhyPred[i][j]<0.5):
                        tn += 1
                    if (pauseTruth[i][j]<0.5 and rhyPred[i][j]>=0.5):
                        fp += 1
                    if (pauseTruth[i][j]>=0.5 and rhyPred[i][j]<0.5):
                        fn += 1
                    if (pauseTruth[i][j]>=0.5 and rhyPred[i][j]>=0.5):
                        tp += 1
    print(f"Total: {cnt}")
    print(f"True Positve: {tp}")
    print(f"False Positive: {fp}")
    print(f"True Negative: {tn}")
    print(f"False Negative: {fn}")



            


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=False,
        default="batch",
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="preprocessed_data/LJSpeech/val.txt",
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )

    parser.add_argument(
        "--restore_step", 
        type=int, 
        default=3000, # just to get it going
        required=False
    )

    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None

    # Read Config
    preprocess_config = yaml.load(open("../Fastspeech2-Original/config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open("config/LJSpeech/model.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open("config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )

    synthesize(model, batchs)
