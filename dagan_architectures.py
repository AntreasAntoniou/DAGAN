import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, layer_norm
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from tensorflow.python.ops.nn_ops import leaky_relu
from network_summary import count_parameters


class UResNetGenerator:
    def __init__(self, layer_sizes, layer_padding, batch_size, num_channels=1,
                 inner_layers=0, name="g"):
        """
        Initialize a UResNet generator.
        :param layer_sizes: A list with the filter sizes for each MultiLayer e.g. [64, 64, 128, 128]
        :param layer_padding: A list with the padding type for each layer e.g. ["SAME", "SAME", "SAME", "SAME"]
        :param batch_size: An integer indicating the batch size
        :param num_channels: An integer indicating the number of input channels
        :param inner_layers: An integer indicating the number of inner layers per MultiLayer
        """
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_sizes = layer_sizes
        self.layer_padding = layer_padding
        self.inner_layers = inner_layers
        self.conv_layer_num = 0
        self.build = True
        self.name = name

    def upscale(self, x, h_size, w_size):
        """
        Upscales an image using nearest neighbour
        :param x: Input image
        :param h_size: Image height size
        :param w_size: Image width size
        :return: Upscaled image
        """
        [b, h, w, c] = [int(dim) for dim in x.get_shape()]

        return tf.image.resize_nearest_neighbor(x, (h_size, w_size))

    def conv_layer(self, inputs, num_filters, filter_size, strides, activation=None,
                   transpose=False, w_size=None, h_size=None):
        """
        Add a convolutional layer to the network.
        :param inputs: Inputs to the conv layer.
        :param num_filters: Num of filters for conv layer.
        :param filter_size: Size of filter.
        :param strides: Stride size.
        :param activation: Conv layer activation.
        :param transpose: Whether to apply upscale before convolution.
        :param w_size: Used only for upscale, w_size to scale to.
        :param h_size: Used only for upscale, h_size to scale to.
        :return: Convolution features
        """
        self.conv_layer_num += 1
        if transpose:
            outputs = self.upscale(inputs, h_size=h_size, w_size=w_size)
            outputs = tf.layers.conv2d_transpose(outputs, num_filters, filter_size,
                                                 strides=strides,
                                       padding="SAME", activation=activation)
        elif not transpose:
            outputs = tf.layers.conv2d(inputs, num_filters, filter_size, strides=strides,
                                                 padding="SAME", activation=activation)
        return outputs

    def remove_duplicates(self, input_features):
        """
        Remove duplicate entries from layer list.
        :param input_features: A list of layers
        :return:
        """
        feature_name_set = set()
        non_duplicate_feature_set = []
        for feature in input_features:
            if feature.name not in feature_name_set:
                non_duplicate_feature_set.append(feature)
            feature_name_set.add(feature.name)
        return non_duplicate_feature_set

    def resize_batch(self, batch_images, size):

        """
        Resize image batch using nearest neighbour
        :param batch_images: Image batch
        :param size: Size to upscale to
        :return: Resized image batch.
        """
        images = tf.image.resize_images(batch_images, size=size, method=ResizeMethod.NEAREST_NEIGHBOR)

        return images

    def add_res_net_encoder_layer(self, input, name, training, dropout_rate, layer_to_skip_connect, local_inner_layers,
                                  num_features, dim_reduce=False):

        """
        Adds a resnet encoder layer.
        :param input: The input to the encoder layer
        :param training: Flag for training or validation
        :param dropout_rate: A float or a placeholder for the dropout rate
        :param layer_to_skip_connect: Layer to skip-connect this layer to.
        :param local_inner_layers: A list with the inner layers of the current Multi-Layer
        :param num_features: Number of feature maps for the convolutions
        :param dim_reduce: Boolean value indicating if this is a
        :return:
        """
        [b1, h1, w1, d1] = input.get_shape().as_list()

        if len(layer_to_skip_connect) >= 2:
            layer_to_skip_connect = layer_to_skip_connect[-2]
        else:
            layer_to_skip_connect = None

        if layer_to_skip_connect is not None:
            [b0, h0, w0, d0] = layer_to_skip_connect.get_shape().as_list()
            if h0 > h1:
                skip_connect_layer = self.conv_layer(layer_to_skip_connect, int(layer_to_skip_connect.get_shape()[3]),
                                                     [1, 1], strides=(2, 2))
            else:
                skip_connect_layer = layer_to_skip_connect
            current_layers = [input, skip_connect_layer]
        else:
            current_layers = [input]

        current_layers.extend(local_inner_layers)
        current_layers = self.remove_duplicates(current_layers)
        outputs = tf.concat(current_layers, axis=3)

        if dim_reduce:
            outputs = self.conv_layer(outputs, num_features, [3, 3], strides=(2, 2))
            outputs = leaky_relu(outputs)
            outputs = batch_norm(outputs, decay=0.99, scale=True,
                                 center=True, is_training=training,
                                 renorm=True)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
        else:
            outputs = self.conv_layer(outputs, num_features, [3, 3], strides=(1, 1))
            outputs = leaky_relu(features=outputs)
            outputs = batch_norm(outputs, decay=0.99, scale=True,
                                 center=True, is_training=training,
                                 renorm=True)

        return outputs

    def add_res_net_decoder_layer(self, input, name, training, dropout_rate, layer_to_skip_connect, local_inner_layers,
                                  num_features, dim_upscale=False, h_size=None, w_size=None):

        """
        Adds a resnet decoder layer.
        :param input: Input features
        :param name: Layer Name
        :param training: Training placeholder or boolean flag
        :param dropout_rate: float placeholder or float indicating the dropout rate
        :param layer_to_skip_connect: Layer to skip connect to.
        :param local_inner_layers: A list with the inner layers of the current MultiLayer
        :param num_features: Num feature maps for convolution
        :param dim_upscale: Dimensionality upscale
        :param h_size: Height to upscale to
        :param w_size: Width to upscale to
        :return:
        """
        [b1, h1, w1, d1] = input.get_shape().as_list()
        if len(layer_to_skip_connect) >= 2:
            layer_to_skip_connect = layer_to_skip_connect[-2]
        else:
            layer_to_skip_connect = None

        if layer_to_skip_connect is not None:
            [b0, h0, w0, d0] = layer_to_skip_connect.get_shape().as_list()

            if h0 < h1:
                skip_connect_layer = self.conv_layer(layer_to_skip_connect,
                                                     int(layer_to_skip_connect.get_shape()[3]),
                                                     [1, 1], strides=(1, 1),
                                                     transpose=True,
                                                     h_size=h_size,
                                                     w_size=w_size)
            else:
                skip_connect_layer = layer_to_skip_connect
            current_layers = [input, skip_connect_layer]
        else:
            current_layers = [input]

        current_layers.extend(local_inner_layers)
        current_layers = self.remove_duplicates(current_layers)
        outputs = tf.concat(current_layers, axis=3)

        if dim_upscale:
            outputs = self.conv_layer(outputs, num_features, [3, 3], strides=(1, 1),
                                      transpose=True, w_size=w_size, h_size=h_size)
            outputs = leaky_relu(features=outputs)
            outputs = batch_norm(outputs,
                                 decay=0.99, scale=True,
                                 center=True, is_training=training,
                                 renorm=True)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
        else:
            outputs = self.conv_layer(outputs, num_features, [3, 3], strides=(1, 1),
                                       transpose=False)
            outputs = leaky_relu(features=outputs)
            outputs = batch_norm(outputs, decay=0.99, scale=True,
                                 center=True, is_training=training,
                                 renorm=True)

        return outputs

    def __call__(self, z_inputs, conditional_input, training=False, dropout_rate=0.0):


        """
        Apply network on data.
        :param z_inputs: Random noise to inject [batch_size, z_dim]
        :param conditional_input: A batch of images to use as conditional [batch_size, height, width, channels]
        :param training: Training placeholder or boolean
        :param dropout_rate: Dropout rate placeholder or float
        :return: Returns x_g (generated images), encoder_layers(encoder features), decoder_layers(decoder features)
        """
        conditional_input = tf.convert_to_tensor(conditional_input)
        with tf.variable_scope(self.name, reuse=self.reuse):
            # reshape from inputs
            outputs = conditional_input
            encoder_layers = []
            current_layers = [outputs]
            with tf.variable_scope('conv_layers'):

                for i, layer_size in enumerate(self.layer_sizes):
                    encoder_inner_layers = [outputs]
                    with tf.variable_scope('g_conv{}'.format(i)):
                        if i==0:
                            outputs = self.conv_layer(outputs, num_filters=64,
                                                      filter_size=(3, 3), strides=(2, 2))
                            outputs = leaky_relu(features=outputs)
                            outputs = batch_norm(outputs, decay=0.99, scale=True,
                                                 center=True, is_training=training,
                                                 renorm=True)
                            current_layers.append(outputs)
                            encoder_inner_layers.append(outputs)
                        else:
                            for j in range(self.inner_layers[i]):
                                outputs = self.add_res_net_encoder_layer(input=outputs,
                                                                        training=training,
                                                                        layer_to_skip_connect=current_layers,
                                                                        num_features=self.layer_sizes[i],
                                                                        dim_reduce=False,
                                                                        local_inner_layers=encoder_inner_layers,
                                                                        dropout_rate=dropout_rate)
                                encoder_inner_layers.append(outputs)
                                current_layers.append(outputs)
                            outputs = self.add_res_net_encoder_layer(input=outputs,
                                training=training, layer_to_skip_connect=current_layers,
                                local_inner_layers=encoder_inner_layers,
                                num_features=self.layer_sizes[i],
                                dim_reduce=True, dropout_rate=dropout_rate)
                            current_layers.append(outputs)
                        encoder_layers.append(outputs)

            g_conv_encoder = outputs

            with tf.variable_scope("vector_expansion"):
                num_filters = 8
                concat_shape = tuple(encoder_layers[-1].get_shape())
                concat_shape = [int(i) for i in concat_shape]
                z_dense_0 = tf.layers.dense(z_inputs, concat_shape[1] * concat_shape[2] * num_filters,
                                            name='input_dense_0')
                z_reshape_0 = tf.reshape(z_dense_0, [self.batch_size, concat_shape[1], concat_shape[2], num_filters],
                                         name='z_reshape_0')
                concat_shape = tuple(encoder_layers[-2].get_shape())
                concat_shape = [int(i) for i in concat_shape]
                z_dense_1 = tf.layers.dense(z_inputs, concat_shape[1] * concat_shape[2] * num_filters,
                                            name='input_dense_1')
                z_reshape_1 = tf.reshape(z_dense_1, [self.batch_size, concat_shape[1], concat_shape[2], num_filters],
                                         name='z_reshape_1')
                num_filters = num_filters / 2
                concat_shape = tuple(encoder_layers[-3].get_shape())
                concat_shape = [int(i) for i in concat_shape]
                z_dense_2 = tf.layers.dense(z_inputs, concat_shape[1] * concat_shape[2] * num_filters,
                                            name='input_dense_2')
                z_reshape_2 = tf.reshape(z_dense_2, [self.batch_size, concat_shape[1] , concat_shape[2], num_filters],
                                         name='z_reshape_2')
                num_filters = num_filters / 2
                concat_shape = tuple(encoder_layers[-4].get_shape())
                concat_shape = [int(i) for i in concat_shape]
                z_dense_3 = tf.layers.dense(z_inputs, concat_shape[1]* concat_shape[2]  * num_filters,
                                            name='input_dense_3')
                z_reshape_3 = tf.reshape(z_dense_3, [self.batch_size, concat_shape[1] , concat_shape[2], num_filters],
                                         name='z_reshape_3')

            z_layers = [z_reshape_0, z_reshape_1, z_reshape_2, z_reshape_3]
            outputs = g_conv_encoder
            decoder_layers = []
            current_layers = [outputs]
            with tf.variable_scope('g_deconv_layers'):
                for i in range(len(self.layer_sizes)+1):
                    if i<3:
                        outputs = tf.concat([z_layers[i], outputs], axis=3)
                        current_layers[-1] = outputs
                    idx = len(self.layer_sizes) - 1 - i
                    num_features = self.layer_sizes[idx]
                    inner_layers = self.inner_layers[idx]
                    upscale_shape = encoder_layers[idx].get_shape().as_list()
                    if idx<0:
                        num_features = self.layer_sizes[0]
                        inner_layers = self.inner_layers[0]
                        outputs = tf.concat([outputs, conditional_input], axis=3)
                        upscale_shape = conditional_input.get_shape().as_list()

                    with tf.variable_scope('g_deconv{}'.format(i)):
                        decoder_inner_layers = [outputs]
                        for j in range(inner_layers):
                            if i==0 and j==0:
                                outputs = self.add_res_net_decoder_layer(input=outputs,
                                                                         name="decoder_inner_conv_{}_{}"
                                                                         .format(i, j),
                                                                         training=training,
                                                                         layer_to_skip_connect=current_layers,
                                                                         num_features=num_features,
                                                                         dim_upscale=False,
                                                                         local_inner_layers=decoder_inner_layers,
                                                                         dropout_rate=dropout_rate)
                                decoder_inner_layers.append(outputs)
                            else:
                                outputs = self.add_res_net_decoder_layer(input=outputs,
                                                                        name="decoder_inner_conv_{}_{}"
                                                                        .format(i, j), training=training,
                                                                             layer_to_skip_connect=current_layers,
                                                                        num_features=num_features,
                                                                        dim_upscale=False,
                                                                        local_inner_layers=decoder_inner_layers,
                                                                        w_size=upscale_shape[1],
                                                                        h_size=upscale_shape[2],
                                                                         dropout_rate=dropout_rate)
                                decoder_inner_layers.append(outputs)
                        current_layers.append(outputs)
                        decoder_layers.append(outputs)

                        if idx>=0:
                            upscale_shape = encoder_layers[idx - 1].get_shape().as_list()
                            if idx == 0:
                                upscale_shape = conditional_input.get_shape().as_list()
                            outputs = self.add_res_net_decoder_layer(
                                input=outputs,
                                name="decoder_outer_conv_{}".format(i),
                                training=training,
                                layer_to_skip_connect=current_layers,
                                num_features=num_features,
                                dim_upscale=True, local_inner_layers=decoder_inner_layers, w_size=upscale_shape[1],
                                h_size=upscale_shape[2], dropout_rate=dropout_rate)
                            current_layers.append(outputs)
                        if (idx-1)>=0:
                            outputs = tf.concat([outputs, encoder_layers[idx-1]], axis=3)
                            current_layers[-1] = outputs

                high_res_layers = []

                for p in range(2):
                    outputs = self.conv_layer(outputs, self.layer_sizes[0], [3, 3], strides=(1, 1),
                                                         transpose=False)
                    outputs = leaky_relu(features=outputs)

                    outputs = batch_norm(outputs,
                                         decay=0.99, scale=True,
                                         center=True, is_training=training,
                                         renorm=True)
                    high_res_layers.append(outputs)
                outputs = self.conv_layer(outputs, self.num_channels, [3, 3], strides=(1, 1),
                                                     transpose=False)
            # output images
            with tf.variable_scope('g_tanh'):
                gan_decoder = tf.tanh(outputs, name='outputs')



        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')

        if self.build:
            print("generator_total_layers", self.conv_layer_num)
            count_parameters(self.variables, name="generator_parameter_num")
        self.build = False
        return gan_decoder, encoder_layers, decoder_layers

class Discriminator:
    def __init__(self, batch_size, layer_sizes, inner_layers, name="d"):
        """
        Initialize a discriminator network.
        :param batch_size: Batch size for discriminator.
        :param layer_sizes: A list with the feature maps for each MultiLayer.
        :param inner_layers: An integer indicating the number of inner layers.
        """
        self.reuse = False
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.inner_layers = inner_layers
        self.conv_layer_num = 0
        self.build = True
        self.name = name

    def upscale(self, x, scale):
        """
            Upscales an image using nearest neighbour
            :param x: Input image
            :param h_size: Image height size
            :param w_size: Image width size
            :return: Upscaled image
        """
        [b, h, w, c] = [int(dim) for dim in x.get_shape()]

        return tf.image.resize_nearest_neighbor(x, (h * scale, w * scale))

    def conv_layer(self, inputs, num_filters, filter_size, strides, activation=None, transpose=False):
        """
        Add a convolutional layer to the network.
        :param inputs: Inputs to the conv layer.
        :param num_filters: Num of filters for conv layer.
        :param filter_size: Size of filter.
        :param strides: Stride size.
        :param activation: Conv layer activation.
        :param transpose: Whether to apply upscale before convolution.
        :return: Convolution features
        """
        self.conv_layer_num += 1
        if transpose:
            outputs = tf.layers.conv2d_transpose(inputs, num_filters, filter_size, strides=strides,
                                       padding="SAME", activation=activation)
        elif not transpose:
            outputs = tf.layers.conv2d(inputs, num_filters, filter_size, strides=strides,
                                                 padding="SAME", activation=activation)
        return outputs

    def remove_duplicates(self, input_features):
        """
        Remove duplicate entries from layer list.
        :param input_features: A list of layers
        :return:
        """
        feature_name_set = set()
        non_duplicate_feature_set = []
        for feature in input_features:
            if feature.name not in feature_name_set:
                non_duplicate_feature_set.append(feature)
            feature_name_set.add(feature.name)
        return non_duplicate_feature_set
    def add_res_net_encoder_layer(self, input, name, training, layer_to_skip_connect, local_inner_layers, num_features,
                                  dim_reduce=False, dropout_rate=0.0):

        """
        :param input:
        :param name:
        :param training:
        :param layer_to_skip_connect:
        :param local_inner_layers:
        :param num_features:
        :param dim_reduce:
        :param dropout_rate:
        :return:
        """
        [b1, h1, w1, d1] = input.get_shape().as_list()
        if layer_to_skip_connect is not None:
            [b0, h0, w0, d0] = layer_to_skip_connect.get_shape().as_list()

            if h0 > h1:
                skip_connect_layer = self.conv_layer(layer_to_skip_connect, int(layer_to_skip_connect.get_shape()[3]),
                                                     [1, 1], strides=(2, 2))
            else:
                skip_connect_layer = layer_to_skip_connect
        else:
            skip_connect_layer = layer_to_skip_connect
        current_layers = [input, skip_connect_layer]
        current_layers.extend(local_inner_layers)
        current_layers = self.remove_duplicates(current_layers)
        outputs = tf.concat(current_layers, axis=3)
        if dim_reduce:
            outputs = self.conv_layer(outputs, num_features, [3, 3], strides=(2, 2))
            outputs = leaky_relu(features=outputs)
            outputs = layer_norm(inputs=outputs, center=True, scale=True)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
        else:
            outputs = self.conv_layer(outputs, num_features, [3, 3], strides=(1, 1))
            outputs = leaky_relu(features=outputs)
            outputs = layer_norm(inputs=outputs, center=True, scale=True)

        return outputs


    def __call__(self, conditional_input, generated_input, training=False, dropout_rate=0.0):
        """
        :param conditional_input: A batch of conditional inputs (x_i) of size [batch_size, height, width, channel]
        :param generated_input: A batch of generated inputs (x_g) of size [batch_size, height, width, channel]
        :param training: Placeholder for training or a boolean indicating training or validation
        :param dropout_rate: A float placeholder for dropout rate or a float indicating the dropout rate
        :param name: Network name
        :return:
        """
        conditional_input = tf.convert_to_tensor(conditional_input)
        generated_input = tf.convert_to_tensor(generated_input)
        with tf.variable_scope(self.name, reuse=self.reuse):
            concat_images = tf.concat([conditional_input, generated_input], axis=3)
            outputs = concat_images
            encoder_layers = []
            current_layers = [outputs]
            with tf.variable_scope('conv_layers'):
                for i, layer_size in enumerate(self.layer_sizes):
                    encoder_inner_layers = [outputs]
                    with tf.variable_scope('g_conv{}'.format(i)):
                        if i == 0:
                            outputs = self.conv_layer(outputs, num_filters=64,
                                                      filter_size=(3, 3), strides=(2, 2))
                            outputs = leaky_relu(features=outputs)
                            outputs = layer_norm(inputs=outputs, center=True, scale=True)
                            current_layers.append(outputs)
                        else:
                            for j in range(self.inner_layers[i]):
                                outputs = self.add_res_net_encoder_layer(input=outputs,
                                                                         name="encoder_inner_conv_{}_{}"
                                                                         .format(i, j), training=training,
                                                                         layer_to_skip_connect=current_layers[-2],
                                                                         num_features=self.layer_sizes[i],
                                                                         dropout_rate=dropout_rate,
                                                                         dim_reduce=False,
                                                                         local_inner_layers=encoder_inner_layers)
                                current_layers.append(outputs)
                            outputs = self.add_res_net_encoder_layer(input=outputs,
                                                                                     name="encoder_outer_conv_{}"
                                                                                     .format(i),
                                                                                     training=training,
                                                                                     layer_to_skip_connect=
                                                                                     current_layers[-2],
                                                                                     local_inner_layers=
                                                                                     encoder_inner_layers,
                                                                                     num_features=self.layer_sizes[i],
                                                                                     dropout_rate=dropout_rate,
                                                                                     dim_reduce=True)
                            current_layers.append(outputs)
                        encoder_layers.append(outputs)


            flatten = tf.contrib.layers.flatten(encoder_layers[-1])
            with tf.variable_scope('discriminator_out'):
                outputs = tf.layers.dense(flatten, 1, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        #view_names_of_variables(self.variables)
        if self.build:
            print("discr layers", self.conv_layer_num)
            count_parameters(self.variables, name="discriminator_parameter_num")
        self.build = False
        return outputs, current_layers