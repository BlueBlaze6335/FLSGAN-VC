import os

import tensorflow as tf
import argparse
from typing import Any
import time
import librosa
from utils.common import Common_helpers
from utils.training_utils import Training_helpers
from utils.inversion import Inversion_helpers
from utils.after_training_utils import After_Training_helpers
import numpy as np
from training.training_loop import train_d, train_all
import matplotlib.pyplot as plt


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def setup_training_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_path", type=str, default=None, required=True,
                        help="Audio file path")
    parser.add_argument("--dest_path", type=str, default='./Test_Results/',
                        help="Destination path to save test results")
    parser.add_argument("--name", type=str, default='./Test_1/',
                        help="The name of test folder")
    parser.add_argument("--format", type=str, default='png',
                        help="The format of spectrogram image")
    parser.add_argument("--model_path", type=str, default=None, required=True,
                        help="Destination path to previously saved network weights")

    parser.add_argument("--hop", type=int, default=192,
                        help="Hop size (window size = 6*hop)")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Sampling rate")
    parser.add_argument("--min_level_db", type=int, default=-100,
                        help="Reference values to normalize data")
    parser.add_argument("--ref_level_db", type=int, default=20,
                        help="Reference values to normalize data")

    parser.add_argument("--shape", type=int, default=24,
                        help="Length of time axis of split spectrograms to feed to generator")
    parser.add_argument("--vec_len", type=int, default=128,
                        help="Length of vector generated by siamese vector")
    parser.add_argument("--device", type=str, default='cpu',
                        help="Device")

    temp_args = parser.parse_args()

    args.file_path = temp_args.file_path
    args.dest_path = temp_args.dest_path
    args.name = temp_args.name
    args.format = temp_args.format
    args.model_path = temp_args.model_path
    args.hop = temp_args.hop
    args.sr = temp_args.sr
    args.min_level_db = temp_args.min_level_db
    args.ref_level_db = temp_args.ref_level_db
    args.shape = temp_args.shape
    args.vec_len = temp_args.vec_len
    args.device = temp_args.device

    return args


if __name__ == "__main__":
    args = EasyDict()
    args = setup_training_args(args)

    TH = Training_helpers(args, aspec=None)

    args.gen, args.critic, args.siam, [args.opt_gen, args.opt_disc] = TH.get_networks(load_model=True,
                                                                                      path=args.model_path)

    wv, sr = librosa.load(args.file_path, sr=args.sr)  # Load waveform

    IH = Inversion_helpers(args)
    ATH = After_Training_helpers(args)

    speca = IH.prep(wv)  # Waveform to Spectrogram
    ATH.towave(speca, name=args.name, path=args.dest_path, format=args.format)  # Convert and save wav
