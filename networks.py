import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Concatenate, Conv2D, Conv2DTranspose, GlobalAveragePooling2D, UpSampling2D, LeakyReLU, ReLU, Add, Multiply, Lambda, Dot, BatchNormalization, Activation, ZeroPadding2D, Cropping2D, Cropping1D
from tensorflow.keras.models import Sequential, Model, load_model
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.eager import context
import tensorflow_addons as tfa

def l2normalize(v, eps=1e-12):
    return v / (tf.norm(v) + eps)


class ConvSN2D(tf.keras.layers.Conv2D):

    def __init__(self, filters, kernel_size, power_iterations=1, **kwargs):
        super(ConvSN2D, self).__init__(filters, kernel_size, **kwargs)
        self.power_iterations = power_iterations

    def build(self, input_shape):
        super(ConvSN2D, self).build(input_shape)

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        self.u = self.add_weight(self.name + '_u',
                                 shape=tuple(
                                     [1, self.kernel.shape.as_list()[-1]]),
                                 initializer=tf.initializers.RandomNormal(
                                     0, 1),
                                 trainable=False
                                 )

    def compute_spectral_norm(self, W, new_u, W_shape):
        for _ in range(self.power_iterations):

            new_v = l2normalize(tf.matmul(new_u, tf.transpose(W)))
            new_u = l2normalize(tf.matmul(new_v, W))

        sigma = tf.matmul(tf.matmul(new_v, W), tf.transpose(new_u))
        W_bar = W/sigma

        with tf.control_dependencies([self.u.assign(new_u)]):
            W_bar = tf.reshape(W_bar, W_shape)

        return W_bar

    def call(self, inputs):
        W_shape = self.kernel.shape.as_list()
        W_reshaped = tf.reshape(self.kernel, (-1, W_shape[-1]))
        new_kernel = self.compute_spectral_norm(W_reshaped, self.u, W_shape)
        outputs = self._convolution_op(inputs, new_kernel)

        if self.use_bias:
            if self.data_format == 'channels_first':
                outputs = tf.nn.bias_add(
                    outputs, self.bias, data_format='NCHW')
            else:
                outputs = tf.nn.bias_add(
                    outputs, self.bias, data_format='NHWC')
        if self.activation is not None:
            return self.activation(outputs)

        return outputs


class ConvSN2DTranspose(tf.keras.layers.Conv2DTranspose):

    def __init__(self, filters, kernel_size, power_iterations=1, **kwargs):
        super(ConvSN2DTranspose, self).__init__(filters, kernel_size, **kwargs)
        self.power_iterations = power_iterations

    def build(self, input_shape):
        super(ConvSN2DTranspose, self).build(input_shape)

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        self.u = self.add_weight(self.name + '_u',
                                 shape=tuple(
                                     [1, self.kernel.shape.as_list()[-1]]),
                                 initializer=tf.initializers.RandomNormal(
                                     0, 1),
                                 trainable=False
                                 )

    def compute_spectral_norm(self, W, new_u, W_shape):
        for _ in range(self.power_iterations):

            new_v = l2normalize(tf.matmul(new_u, tf.transpose(W)))
            new_u = l2normalize(tf.matmul(new_v, W))

        sigma = tf.matmul(tf.matmul(new_v, W), tf.transpose(new_u))
        W_bar = W/sigma

        with tf.control_dependencies([self.u.assign(new_u)]):
            W_bar = tf.reshape(W_bar, W_shape)

        return W_bar

    def call(self, inputs):
        W_shape = self.kernel.shape.as_list()
        W_reshaped = tf.reshape(self.kernel, (-1, W_shape[-1]))
        new_kernel = self.compute_spectral_norm(W_reshaped, self.u, W_shape)

        inputs_shape = array_ops.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height, width = inputs_shape[h_axis], inputs_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        out_height = conv_utils.deconv_output_length(height,
                                                     kernel_h,
                                                     padding=self.padding,
                                                     output_padding=out_pad_h,
                                                     stride=stride_h,
                                                     dilation=self.dilation_rate[0])
        out_width = conv_utils.deconv_output_length(width,
                                                    kernel_w,
                                                    padding=self.padding,
                                                    output_padding=out_pad_w,
                                                    stride=stride_w,
                                                    dilation=self.dilation_rate[1])
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        output_shape_tensor = array_ops.stack(output_shape)
        outputs = K.conv2d_transpose(
            inputs,
            new_kernel,
            output_shape_tensor,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if not context.executing_eagerly():
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        if self.use_bias:
            outputs = tf.nn.bias_add(
                outputs,
                self.bias,
                data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class DenseSN(Dense):
    def build(self, input_shape):
        super(DenseSN, self).build(input_shape)

        self.u = self.add_weight(self.name + '_u',
                                 shape=tuple(
                                     [1, self.kernel.shape.as_list()[-1]]),
                                 initializer=tf.initializers.RandomNormal(
                                     0, 1),
                                 trainable=False)

    def compute_spectral_norm(self, W, new_u, W_shape):
        new_v = l2normalize(tf.matmul(new_u, tf.transpose(W)))
        new_u = l2normalize(tf.matmul(new_v, W))
        sigma = tf.matmul(tf.matmul(new_v, W), tf.transpose(new_u))
        W_bar = W/sigma
        with tf.control_dependencies([self.u.assign(new_u)]):
            W_bar = tf.reshape(W_bar, W_shape)
        return W_bar

    def call(self, inputs):
        W_shape = self.kernel.shape.as_list()
        W_reshaped = tf.reshape(self.kernel, (-1, W_shape[-1]))
        new_kernel = self.compute_spectral_norm(W_reshaped, self.u, W_shape)
        rank = len(inputs.shape)
        if rank > 2:
            outputs = standard_ops.tensordot(
                inputs, new_kernel, [[rank - 1], [0]])
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            inputs = math_ops.cast(inputs, self._compute_dtype)
            if K.is_sparse(inputs):
                outputs = sparse_ops.sparse_tensor_dense_matmul(
                    inputs, new_kernel)
            else:
                outputs = gen_math_ops.mat_mul(inputs, new_kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


init = tf.keras.initializers.he_uniform()


def conv2d(layer_input, filters, kernel_size=4, strides=2, padding='same', leaky=True, bnorm=True, sn=True):
    if leaky:
        Activ = LeakyReLU(alpha=0.2)
    else:
        Activ = ReLU()
    if sn:
        d = ConvSN2D(filters, kernel_size=kernel_size, strides=strides,
                     padding=padding, kernel_initializer=init, use_bias=False)(layer_input)
    else:
        d = Conv2D(filters, kernel_size=kernel_size, strides=strides,
                   padding=padding, kernel_initializer=init, use_bias=False)(layer_input)
    if bnorm:
        d = BatchNormalization()(d)
    d = Activ(d)
    return d


def deconv2d(layer_input, layer_res, filters, kernel_size=4, conc=True, scalev=False, bnorm=True, up=True, padding='same', strides=2):
    if up:
        u = UpSampling2D((1, 2))(layer_input)
        u = ConvSN2D(filters, kernel_size, strides=(
            1, 1), kernel_initializer=init, use_bias=False, padding=padding)(u)
    else:
        u = ConvSN2DTranspose(filters, kernel_size, strides=strides,
                              kernel_initializer=init, use_bias=False, padding=padding)(layer_input)
    if bnorm:
        u = BatchNormalization()(u)
    u = LeakyReLU(alpha=0.2)(u)
    if conc:
        u = Concatenate()([u, layer_res])
    return u


# def downsample(filters, size, apply_batchnorm=True, padding=False, stride=2, pool=False):
#     initializer = tf.random_normal_initializer(0., 0.02)
#     if padding:
#         result = tf.keras.Sequential()
#         result.add(
#             tf.keras.layers.Conv2D(filters, size, strides=stride, padding="valid",
#                                    kernel_initializer=initializer, use_bias=False))
#     else:
#         result = tf.keras.Sequential()
#         result.add(
#             tf.keras.layers.Conv2D(filters, size, strides=stride, padding="same",
#                                    kernel_initializer=initializer, use_bias=False))
#     if pool:
#         result.add(tf.keras.layers.AveragePooling2D(
#             pool_size=(2, 2), padding="valid"))
#
#     if apply_batchnorm:
#         result.add(tf.keras.layers.BatchNormalization())
#
#     result.add(tf.keras.layers.LeakyReLU())
#
#     return result

def downsample(filters, size, apply_batchnorm=True, padding=False, stride=2, pool=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    if padding:
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=stride, padding="valid",
                                   kernel_initializer=initializer, use_bias=False))
    else:
        result = tf.keras.Sequential()
        result.add(
            ConvSN2D(filters, size, strides=stride, padding="same",
                     kernel_initializer=initializer, use_bias=False))
    if pool:
        result.add(tf.keras.layers.AveragePooling2D(
            pool_size=(1, 1), padding="valid"))

    if apply_batchnorm:
        result.add(tfa.layers.InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

# def downsample(filters, size, apply_batchnorm=True, padding=False, stride=2, pool=False, rate=(1, 1)):
#     initializer = tf.random_normal_initializer(0., 0.02)
#     if padding:
#         result = tf.keras.Sequential()
#         result.add(
#             tf.keras.layers.Conv2D(filters, size, strides=stride, padding="valid", dilation_rate=rate,
#                                    kernel_initializer=initializer, use_bias=False))
#     else:
#         result = tf.keras.Sequential()
#         result.add(
#             tf.keras.layers.Conv2D(filters, size, strides=stride, padding="same",
#                                    kernel_initializer=initializer, use_bias=False))
#     if pool:
#         result.add(tf.keras.layers.AveragePooling2D(
#             pool_size=(1, 1), padding="valid"))
#
#     if apply_batchnorm:
#         result.add(tf.keras.layers.BatchNormalization())
#
#     result.add(tf.keras.layers.LeakyReLU())
#
#     return result

# def upsample(filters, size, apply_dropout=False):
#   initializer = tf.random_normal_initializer(0., 0.02)
#
#   result = tf.keras.Sequential()
#   result.add(
#     tf.keras.layers.Conv2DTranspose(filters, size, strides=[2,2],
#                                     padding='same',
#                                     kernel_initializer=initializer,
#                                     use_bias=False))
#
#   result.add(tf.keras.layers.BatchNormalization())
#
#   if apply_dropout:
#       result.add(tf.keras.layers.Dropout(0.5))
#
#   result.add(tf.keras.layers.ReLU())
#
#   return result

def upsample(filters, size, strides=[2, 2], apply_dropout=False, pool=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        ConvSN2DTranspose(filters, size, strides=strides,
                          padding='same',
                          kernel_initializer=initializer,
                          use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    if pool:
        result.add(tf.keras.layers.AveragePooling2D(
            pool_size=(1, 1), padding="valid"))

    return result

# def upsample(filters, size, strides=[2, 2], apply_dropout=False, pool=False, rate=(1, 1)):
#     initializer = tf.random_normal_initializer(0., 0.02)
#
#     result = tf.keras.Sequential()
#     result.add(
#         tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,
#                                         padding='same',
#                                         dilation_rate=rate,
#                                         kernel_initializer=initializer,
#                                         use_bias=False))
#
#     result.add(tf.keras.layers.BatchNormalization())
#
#     if apply_dropout:
#         result.add(tf.keras.layers.Dropout(0.5))
#
#     result.add(tf.keras.layers.LeakyReLU())
#     if pool:
#         result.add(tf.keras.layers.AveragePooling2D(
#             pool_size=(1, 1), padding="valid"))
#
#     return result


def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

#Attention Layer
def attention(inp):
  tf.print(inp.shape)
  k=Conv2D(filters=1, kernel_size=1,strides=1, kernel_initializer='glorot_uniform')(inp)
  tf.print(k.shape)
  q=Conv2D(filters=1, kernel_size=1,strides=1, kernel_initializer='glorot_uniform')(inp)
  tf.print(q.shape)
  v=Conv2D(filters=1,kernel_size=1,strides=1, kernel_initializer='glorot_uniform')(inp)
  tf.print(v.shape)
  s=tf.linalg.matmul(q,v, transpose_b=True)
  tf.print(s.shape)
  beta=tf.nn.softmax(s)
  tf.print(beta.shape)
  o= tf.linalg.matmul(beta, k)
  gamma= tf.Variable(name="gamma",initial_value=[1.0])
  #o=tf.reshape(o, shape=input_shape)
  O=Conv2D(input_shape=o.shape, filters=1,kernel_size=1, strides=1)(o)
  x=tf.math.multiply(gamma,O)+inp
  tf.print(x.shape)
  return x


# Extract function: splitting spectrograms
def extract_image(im):
  im1 = Cropping2D(((0,0), (0, 2*(im.shape[2]//3))))(im)
  im2 = Cropping2D(((0,0), (im.shape[2]//3,im.shape[2]//3)))(im)
  im3 = Cropping2D(((0,0), (2*(im.shape[2]//3), 0)))(im)
  return im1,im2,im3

# Assemble function: concatenating spectrograms
def assemble_image(lsim):
  im1,im2,im3 = lsim
  imh = Concatenate(2)([im1,im2,im3])
  return imh


# -------------------------------------------------------------------------------------------------------------------------------------
# Networks Architecture

# def Generator(input_shape):
#     h, w, c = input_shape
#     inputs = Input(shape=input_shape)
#
#     down_stack = [
#         downsample(4, [5, 5], padding=True, stride=2, rate=(2, 2)),  # (batch_size, 1, 24, 64)
#         downsample(16, [9, 9], stride=[2, 2], rate=(2, 2)),  # (batch_size, 1, 12, 128)
#         downsample(64, [9, 9], stride=[2, 2], rate=(2, 2)),  # (batch_size, 1, 6, 256)
#         downsample(256, [7, 7], stride=[2, 2], rate=(2, 2)),  # (batch_size, 1, 3, 512)
#     ]
#
#     up_stack = [
#         upsample(128, [5, 5], strides=[1,1]),  # (batch_size, 1, 3, 512)
#         upsample(64, [7, 7], strides=[2,5], ),  # (batch_size, 4, 6, 256)
#         upsample(16, [9, 9],strides=[2,2] ),  # (batch_size, 8, 12, 1024)
#         upsample(4, [9, 12], strides=[4,2]),
#     ]
#
#     initializer = tf.random_normal_initializer(0., 0.02)
#     last = tf.keras.layers.Conv2DTranspose(1, (1, 1),
#                                            strides=(1, 1),
#                                            padding='valid',
#                                            kernel_initializer=initializer,
#                                            activation='tanh')  # (batch_size, 256, 256, 3)
#     # 1, kernel_size=(h,1), strides=(1,1), kernel_initializer=init, padding='valid', activation='tanh'
#     x = inputs
#
#     # Downsampling through the model
#     skips = []
#     #x = tf.keras.layers.ZeroPadding2D((0, 1))(x)  # (26,1)
#     for down in down_stack:
#         x = down(x)
#         skips.append(x)
#     skips = reversed(skips[:-1])
#     x = attention(x)
#     # Upsampling and establishing the skip connections
#     for up in up_stack:
#         x = up(x)
#         # x = tf.keras.layers.Concatenate()([x, skip])
#
#     x = last(x)  # tf.keras.activations.tanh(x)
#     model = tf.keras.Model(inputs=inputs, outputs=x, name="Generator")
#     model.summary()
#     # last_down=skips[0]
#     # feature=Flatten()(last_down)
#     # tf.print(feature)
#     # last_down = tf.reduce_mean(last_down)
#     return model

def Generator(input_shape):
    h, w, c = input_shape
    inputs = Input(shape=input_shape)

    down_stack = [
        downsample(4, [5, 5], padding=True, stride=1),  # (batch_size, 1, 24, 64)
        downsample(16, [9, 9], stride=[2, 2]),  # (batch_size, 1, 12, 128)
        downsample(64, [9, 9], stride=[2, 2]),  # (batch_size, 1, 6, 256)
        downsample(256, [7, 7], stride=[2, 2]),  # (batch_size, 1, 3, 512)
    ]

    up_stack = [
        upsample(64, [5, 5], apply_dropout=True),  # (batch_size, 1, 3, 512)
        upsample(16, [7, 7], apply_dropout=True),  # (batch_size, 4, 6, 256)
        upsample(4, [9, 9]),  # (batch_size, 8, 12, 1024)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, (1, 1),
                                           strides=1,
                                           padding='valid',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 256, 256, 3)
    # 1, kernel_size=(h,1), strides=(1,1), kernel_initializer=init, padding='valid', activation='tanh'
    x = inputs

    # Downsampling through the model
    skips = []
    x = tf.keras.layers.ZeroPadding2D((0, 1))(x)  # (26,1)
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    x = attention(x)
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        # x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    model = tf.keras.Model(inputs=inputs, outputs=x, name="Generator")
    model.summary()
    # last_down=skips[0]
    # feature=Flatten()(last_down)
    # tf.print(feature)
    # last_down = tf.reduce_mean(last_down)
    return model

# def Discriminator(inp_shape):
#   initializer = tf.random_normal_initializer(0., 0.02)
#
#   inp = tf.keras.layers.Input(shape=inp_shape, name='input_image')
#
#   #x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)
#
#   down1 = downsample(64, 4, False)(inp)  # (batch_size, 128, 128, 64)
#   down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
#   down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)
#
#   zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
#   conv = tf.keras.layers.Conv2D(256, 4, strides=1,
#                                 kernel_initializer=initializer,
#                                 use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)
#
#   batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
#
#   leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
#
#   zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)
#
#   last = tf.keras.layers.Conv2D(256, 4, strides=1,
#                                 kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)
#   flatten=Flatten()(last)
#   dense = Dense(1)(flatten)
#
#   model = tf.keras.Model(inputs=inp, outputs=dense , name="Discriminator")
#   model.summary()
#   return model

def Discriminator(inp_shape):
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=inp_shape, name='input_image')
  tar = tf.keras.layers.Input(shape=inp_shape, name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  model = tf.keras.Model(inputs=[inp, tar], outputs=last , name="Discriminator")
  model.summary()
  return model

# U-NET style architecture
# def build_siamese(input_shape):
#   h,w,c = input_shape
#   inp = Input(shape=input_shape)
#   g1 = conv2d(inp, 64, kernel_size=(4,4), strides=(2,2), padding='valid', sn=False)
#   g2 = conv2d(g1, 128, kernel_size=(4,4), strides=(2,2), sn=False)
#   g3 = conv2d(g2, 256, kernel_size=(4,4), strides=(2,2), sn=False)
#   #g4 = attention(g3)
#   g5 = Flatten()(g3)
#   g5 = Dense(128)(g5)
#   model=Model(inp,g5,name="Siamese")
#   model.summary()
#   return model

def build_siamese(input_shape):
  h,w,c = input_shape
  inp = Input(shape=input_shape)
  g1 = conv2d(inp, 64, kernel_size=(4,4), strides=(2,2), padding='valid', sn=False)
  g2 = conv2d(g1, 128, kernel_size=(4,4), strides=(2,2), sn=False)
  g3 = conv2d(g2, 128, kernel_size=(4,4), strides=(2,2), sn=False)
  #g4 = attention(g3)
  g5 = Flatten()(g3)
  g5 = Dense(128)(g5)
  model=Model(inp,g5,name="Siamese")
  model.summary()
  return model

def build_critic(input_shape):
  h,w,c = input_shape
  inp = Input(shape=input_shape)
  #g0 = tf.keras.layers.ZeroPadding2D((1,0))(inp)
  g1 = conv2d(inp, 32, kernel_size=(8,8), strides=(2,3), padding='valid', bnorm=False)
  g2 = conv2d(g1, 64, kernel_size=(6,6), strides=(2,2), bnorm=False)
  g3 = conv2d(g2, 128, kernel_size=(6,6), strides=(2,2), bnorm=False)
  g4 = conv2d(g3, 256, kernel_size=(5,5), strides=(1,2), bnorm=False)
  #g5 = attention(g4)
  g6=Flatten()(g4)
  g7 = DenseSN(1, kernel_initializer=init)(g6)
  model=Model(inp,g7,name="Critic")
  model.summary()
  return model