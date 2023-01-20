import tensorflow as tf

n = 4096  # Modulation Spectrum FFT length
n = tf.convert_to_tensor(4096)
n = tf.expand_dims(n,axis=-1)
# print(n)

# Calculate MCEP
def get_mcep(data):
    data = tf.squeeze(data)  # Squeeze data
    fft_mag = tf.math.abs(data)  # taking absolute values
    fft_mag = tf.clip_by_value(fft_mag, 0.000001, fft_mag.dtype.max)  # removing null values and replacing by 0.000001
    mc = tf.signal.dct(fft_mag, n=24)  # direct cosine transform (dim =24)
    return mc

# Calculate Modulation Spectra
def modspec(x, n=n, norm=None, return_phase=False):
    # print(x.shape)
    # DFT against time axis
    x = tf.reduce_mean(x, axis=(0,), keepdims=True)  # taking mean of mcep values
    x = tf.squeeze(x)
    x = tf.transpose(x) # transposing to perform rfft over the coefficients of the batch
    s = tf.signal.rfft(x, fft_length=n)  # rfft
    s_complex = tf.signal.fft(s)  # fft to 1D n-point discrete Fourier Transform (DFT) [equivalent to np.fft.rfft]
    s_complex = tf.transpose(s_complex)  # transposing again to get the original structure
    R, im = tf.math.real(s_complex), tf.math.imag(s_complex)  # separating the real and imaginary values
    ms = tf.math.multiply(R, R) + tf.math.multiply(im, im)  # square sum of real and imaginary values
    return ms

# Mean of Modulation Spectra
def mean_modspec(data):
    mgc = get_mcep(data)  # calculate mcep
    lg = tf.math.log(modspec(mgc, n=n))  # log of modulation spectra
    ms = tf.expand_dims(lg, axis=-1)
    ms = tf.math.reduce_mean(ms, axis=(2,), keepdims=True)  # mean modulation spectrum
    ms = tf.squeeze(ms)
    return ms  # np.mean(np.array(mss), axis=(0,))


# MSD loss
def msd_loss(ori, fake):
    new = 0
    ori = mean_modspec(ori)  # mean modspec for original audio
    fake = mean_modspec(fake)  # mean modspec of fake audio
    # calculating difference across each dimension (24)
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

