import tensorflow as tf

n = 4096
n = tf.convert_to_tensor(4096)
n = tf.expand_dims(n,axis=-1)
# print(n)

def get_mcep(data):
    data = tf.squeeze(data)  # np.squeeze(data)
    # data=tf.slice(data,(1),(1,1))
    # print(data.shape)
    fft_mag = tf.math.abs(data)
    # print(fft_mag.shape, " fftmag shape")
    fft_mag = tf.clip_by_value(fft_mag, 0.000001, fft_mag.dtype.max)
    # fft_mag[fft_mag == 0] = tf.keras.backend.epsilon()
    # plt.imshow((np.array(fft_mag))[0],cmap=None)
    mc = tf.signal.dct(fft_mag, n=24)  # pysptk.sp2mc(data, order=24, alpha=alpha)
    # c0, mc = mc[:, 0], mc[:, 1:]
    # print(mc.shape, "mc shape")
    return mc


def modspec(x, n=n, norm=None, return_phase=False):
    # print(x.shape)
    # DFT against time axis
    x = tf.reduce_mean(x, axis=(0,), keepdims=True)
    x = tf.squeeze(x)
    x = tf.transpose(x)
    #x = tf.transpose(x[0, :, :])
    # print(x.shape, "x transpose shape")
    s = tf.signal.rfft(x, fft_length=n)
    # print(s.shape, "shape rfft")
    s_complex = tf.signal.fft(s)
    s_complex = tf.transpose(s_complex)
    # print(s_complex.shape, "s_complex shape")
    # assert s_complex.shape[0] == n // 2 + 1
    R, im = tf.math.real(s_complex), tf.math.imag(s_complex)
    ms = tf.math.multiply(R, R) + tf.math.multiply(im, im)
    # print(ms.shape, "ms shape")
    return ms


def mean_modspec(data):
    # mss=[]
    mgc = get_mcep(data)
    lg = tf.math.log(modspec(mgc, n=n))
    # print(lg.shape, "log shape")
    ms = tf.expand_dims(lg, axis=-1)
    # print(ms.shape, "check shape log")
    ms = tf.math.reduce_mean(ms, axis=(2,), keepdims=True)  # np.log(modspec(mgc, n=4096))
    # mss.append(ms)
    ms = tf.squeeze(ms)
    # print(ms.shape, "mean ms shape")
    return ms  # np.mean(np.array(mss), axis=(0,))


# MSD loss
def msd_loss(ori, fake):
    # print(ori[:,:,:,:].shape)
    new = 0
    ori = mean_modspec(ori)
    fake = mean_modspec(fake)
    #print(ori.shape, "ori shape")
    for i in range(24):
        a = tf.transpose(ori[i, :])  # ori[i, :].T
        b = tf.transpose(fake[i, :])  # fake[i,:].T
        # print(a.shape,"a shape")
        diff = tf.math.reduce_mean(tf.math.abs((a - b)), axis=None, keepdims=True)  # np.mean(np.absolute(a-b))
        diff = tf.tensordot(diff, diff, 1)  # (np.inner(diff, diff))
        new = new + diff

    MSD = tf.math.sqrt(1 / 24) * tf.math.sqrt(new)
    # tf.print(MSD)
    return MSD


# def modspec(x, n=4096, norm=None, return_phase=False):
#     # print(x.shape)
#     # DFT against time axis
#     s_complex = np.fft.rfft(x, n=n, axis=0, norm=norm)
#     # print(s_complex.shape,"s_complex shape")
#     assert s_complex.shape[0] == n // 2 + 1
#     R, im = s_complex.real, s_complex.imag
#     ms = R * R + im * im
#     # print(ms.shape,"ms shape")
#     # TODO: this is ugly...
#     if return_phase:
#         return ms, np.exp(1.0j * np.angle(s_complex))
#     else:
#         return ms
#
#
#
# def mean_modspec(data):
#     mss = []
#     mgc = get_mcep(data)
#     ms = np.log(modspec(mgc, n=4096))
#     # print(ms.shape,"log shape")
#     mss.append(ms)
#     ms = np.mean(np.array(mss), axis=(0,))
#     # print(ms.shape,"mean ms shape")
#     # print((np.mean(np.array(mss), axis=(0,)).shape))
#     return ms  # np.mean(np.array(mss), axis=(0,))
#
#
# def msd_loss(ori, fake):
#     # print(ori[:,:,:,:].shape)
#     new = 0
#     ori = mean_modspec(ori)
#     fake = mean_modspec(fake)
#     # print(ori.shape,"ori shape")
#     for i in range(24):
#         a = ori[i, :].T
#         b = fake[i, :].T
#         # print(a.shape,"a shape")
#         diff = np.mean(np.absolute(a - b))
#         diff = (np.inner(diff, diff))
#         new = new + diff
#
#     MSD = math.sqrt(1 / 24) * math.sqrt(new)
#     return tf.convert_to_tensor(MSD)



