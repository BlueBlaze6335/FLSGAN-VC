import tensorflow as tf
import argparse
from typing import Any
import time
from utils.common import Common_helpers
from utils.training_utils import Training_helpers
import numpy as np
from training.training_loop import train_d, train_all
import math
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

    parser.add_argument("--awv_path", type=str, default='/media/nd/disk3/Neel/audio_style_transfer/Data/cmu_us_bdl_arctic/wav',
                        help="Source spectrograms path")
    parser.add_argument("--bwv_path", type=str, default='/media/nd/disk3/Neel/audio_style_transfer/Data/cmu_us_clb_arctic/wav',
                        help="Target spectrograms path")
    parser.add_argument("--dest_path", type=str, default='/home/nd/Desktop/Results_melgan',
                        help="Destination path to save network weights")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Destination path to previously saved network weights")

    parser.add_argument("--id_loss_weight", type=float, default=1.5,
                        help="Weight for id loss")
    parser.add_argument("--travel_loss_weight", type=float, default=10.,
                        help="Weight for travel loss")

    parser.add_argument("--hop", type=int, default=192,
                        help="Hop size (window size = 6*hop)")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Sampling rate")
    parser.add_argument("--min_level_db", type=int, default=-150,
                        help="Reference values to normalize data")
    parser.add_argument("--ref_level_db", type=int, default=50,
                        help="Reference values to normalize data")

    parser.add_argument("--shape", type=int, default=24,
                        help="Length of time axis of split spectrograms to feed to generator")
    parser.add_argument("--vec_len", type=int, default=128,
                        help="Length of vector generated by siamese vector")
    parser.add_argument("--batch_size", type=int, default=100,help="Batch size")
    parser.add_argument("--delta", type=float, default=2.,
                        help="Constant for siamese loss")
    parser.add_argument("-lr", type=float, default=0.0002,
                        help="Learning rate")

    parser.add_argument("--n_save", type=int, default=10,
                        help="How many epochs between each saving and displaying of results")
    parser.add_argument("--gupt", type=int, default=1,
                        help="How many discriminator updates for generator+siamese update")
    parser.add_argument("--epoch", type=int, default=201,
                        help="Epoch number")
    parser.add_argument("--device", type=str, default='gpu',
                        help="Device")

    temp_args = parser.parse_args()

    args.awv_path = temp_args.awv_path
    args.bwv_path = temp_args.bwv_path
    args.dest_path = temp_args.dest_path
    args.model_path = temp_args.model_path

    args.id_loss_weight = temp_args.id_loss_weight
    args.travel_loss_weight = temp_args.travel_loss_weight

    args.hop = temp_args.hop
    args.sr = temp_args.sr
    args.min_level_db = temp_args.min_level_db
    args.ref_level_db = temp_args.ref_level_db

    args.shape = temp_args.shape
    args.vec_len = temp_args.vec_len
    args.batch_size = temp_args.batch_size
    args.delta = temp_args.delta
    args.lr = temp_args.lr

    args.n_save = temp_args.n_save
    args.gupt = temp_args.gupt
    args.epoch = temp_args.epoch
    args.device = temp_args.device

    return args


if __name__ == "__main__":

    args = EasyDict()
    args = setup_training_args(args)

    CH = Common_helpers(args)
    # MALE1
    awv = CH.audio_array(args.awv_path)  # get waveform array from folder containing wav files
    aspec = CH.tospec(awv)  # get spectrogram array
    adata = CH.splitcut(aspec)  # split spectrogams to fixed length
    # FEMALE1
    bwv = CH.audio_array(args.bwv_path)
    bspec = CH.tospec(bwv)
    bdata = CH.splitcut(bspec)


    @tf.function
    def proc(x):
        return tf.image.random_crop(x, size=[args.hop, 3 * args.shape, 1])


    dsa = tf.data.Dataset.from_tensor_slices(adata).repeat(50).map(
        proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(args.batch_size,
                                                                                     drop_remainder=True)
    dsb = tf.data.Dataset.from_tensor_slices(bdata).repeat(50).map(
        proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(args.batch_size,
                                                                                     drop_remainder=True)

    # Build models and initialize optimizers
    # If load_model=True, specify the path where the models are saved

    TH = Training_helpers(args, aspec , bspec)

    if args.model_path is None:
        args.gen, args.critic, args.siam, [args.opt_gen, args.opt_disc] = TH.get_networks(load_model=False)
    else:
        args.gen, args.critic, args.siam, [args.opt_gen, args.opt_disc] = TH.get_networks(load_model=True,
                                                                                          path=args.model_path)

    TH.update_lr(args.lr)
    df_list = []
    dr_list = []
    g_list = []
    id_list = []
    msd_list=[]
    c = 0
    g = 0

    for epoch in range(args.epoch):
        bef = time.time()
        # args.lr=args.lr*(math.exp(-(0.1)*epoch))
        # TH.update_lr(args.lr)
        for batchi, (a, b) in enumerate(zip(dsa, dsb)):

            if batchi % args.gupt == 0:
                drloss, dfloss, gloss, idloss , msdloss= train_all(a, b, args)
            else:
                drloss,dfloss = train_d(a, b, args)

            df_list.append(dfloss)
            dr_list.append(drloss)
            g_list.append(gloss)
            id_list.append(idloss)
            msd_list.append(msdloss)
            c += 1
            g += 1

            if batchi % 500 == 0:
                print(f'[Epoch {epoch}/{args.epoch}] [Batch {batchi}] [D loss f : {np.mean(df_list[-g:], axis=0)} ',end='')
                print(f'r: {np.mean(dr_list[-g:], axis=0)}] ', end='')
                print(f'[G loss: {np.mean(g_list[-g:], axis=0)}] ', end='')
                print(f'[ID loss: {np.mean(id_list[-g:])}] ', end='')
                print(f'[MSD loss: {np.mean(msd_list[-g:])}] ', end='')
                print(f'[LR: {args.lr}]')
                g = 0
            nbatch = batchi

        print(f'Time - {(time.time() - bef)} s')
        TH.save_end(epoch, np.mean(g_list[-args.n_save * c:], axis=0), np.mean(
            df_list[-args.n_save * c:], axis=0), np.mean(id_list[-args.n_save * c:], axis=0), n_save=args.n_save,
                    save_path=args.dest_path)
        print(
            f'Mean D loss: {np.mean(df_list[-c:], axis=0)} Mean G loss: {np.mean(g_list[-c:], axis=0)} Mean ID loss: {np.mean(id_list[-c:], axis=0)} Mean MSD loss: {np.mean(msd_list[-c:], axis=0)}')
        c = 0


print("End")