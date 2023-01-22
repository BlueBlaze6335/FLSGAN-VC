from __future__ import print_function, division
import tensorflow as tf

from training.networks import extract_image, assemble_image
from training.loss import Losses
from MSD import msd_loss

# Training Functions
gamma = 10
bs = 100
# Train Generator, Siamese and Critic
@tf.function
def train_all(a, b , args):
    ori = b
    # print(tf.reshape(ori,[2,1]).shape)
    # splitting spectrogram in 3 parts
    aa, aa2, aa3 = extract_image(a)
    bb, bb2, bb3 = extract_image(b)
    # print(b," ori type")
    Loss = Losses(args.delta)
    with tf.GradientTape() as tape_gen, tf.GradientTape() as tape_disc:
        # translating A to B
        fab = args.gen(aa, training=True)

        fab2 = args.gen(aa2, training=True)

        fab3 = args.gen(aa3, training=True)

        # identity mapping B to B                                                        COMMENT THESE 3 LINES IF THE IDENTITY LOSS TERM IS NOT NEEDED
        fid = args.gen(bb, training=True)
        fid2 = args.gen(bb2, training=True)
        fid3 = args.gen(bb3, training=True)
        # concatenate/assemble converted spectrograms
        fabtot = assemble_image([fab, fab2, fab3])
        fake = fabtot

        # feed concatenated spectrograms to critic
        cab = args.critic(fabtot, training=True)
        cb = args.critic(b, training=True)
        # feed 2 pairs (A,G(A)) extracted spectrograms to Siamese
        sab = args.siam(fab, training=True)
        sab2 = args.siam(fab2, training=True)
        #sab3 = args.siam(fab3, training=True)
        sa = args.siam(aa, training=True)
        sa2 = args.siam(aa2, training=True)
        #sa3 = args.siam(aa3, training=True)

        # identity mapping loss
        loss_id = (Loss.mae(bb, fid) + Loss.mae(bb2, fid2) + Loss.mae(bb3,
                                                       fid3)) / 3.  # loss_id = 0. IF THE IDENTITY LOSS TERM IS NOT NEEDED
        # travel loss
        loss_m = Loss.loss_travel(sa, sab, sa2, sab2) + Loss.loss_siamese(sa, sa2)
        # msd loss
        loss_msd = msd_loss(ori, fake)
        # generator and critic losses
        loss_g = Loss.g_loss_f(cab)
        loss_dr = Loss.d_loss_r(cb)
        loss_df = Loss.d_loss_f(cab)
        loss_d = (loss_dr + loss_df) / 2.
        # generator+siamese total loss
        # disc_loss=discriminator_loss(cb,cab)
        lossgtot = loss_g + 10. * loss_m + 0.5 * loss_id +0.*loss_msd  # CHANGE LOSS WEIGHTS HERE  (COMMENT OUT +w*loss_id IF THE IDENTITY LOSS TERM IS NOT NEEDED)
        # tf.print(lossgtot)

    # print(fabtot.shape," fabtot ")
    # computing and applying gradients
    grad_gen = tape_gen.gradient(lossgtot, args.gen.trainable_variables + args.siam.trainable_variables)
    args.opt_gen.apply_gradients(zip(grad_gen, args.gen.trainable_variables + args.siam.trainable_variables))

    grad_disc = tape_disc.gradient(loss_d, args.critic.trainable_variables)
    args.opt_disc.apply_gradients(zip(grad_disc, args.critic.trainable_variables))

    return loss_dr, loss_df, loss_g, loss_id, loss_msd


# Train Critic only
@tf.function
def train_d(a, b , args):
    Loss = Losses(args.delta)
    aa, aa2, aa3 = extract_image(a)
    with tf.GradientTape() as tape_disc:
        fab = args.gen(aa, training=True)
        fab2 = args.gen(aa2, training=True)
        fab3 = args.gen(aa3, training=True)
        fabtot = assemble_image([fab, fab2, fab3])

        #     cab = critic(fabtot, training=True)
        #     cb = critic(b, training=True)

        cab = args.critic(fabtot, training=True)
        cb = args.critic(b, training=True)

        loss_dr = Loss.d_loss_r(cb)
        loss_df = Loss.d_loss_f(cab)

        loss_d = (loss_dr + loss_df) / 2.

        # disc_loss=discriminator_loss(cb,cab)

    grad_disc = tape_disc.gradient(loss_d, args.critic.trainable_variables)
    args.opt_disc.apply_gradients(zip(grad_disc, args.critic.trainable_variables))

    return loss_dr, loss_df