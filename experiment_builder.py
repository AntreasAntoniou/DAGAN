import utils.interpolations as interpolations
import numpy as np
import tqdm
from utils.storage import save_statistics, build_experiment_folder
from tensorflow.contrib import slim

from dagan_networks_wgan import *
from utils.sampling import sample_generator, sample_two_dimensions_generator


class ExperimentBuilder(object):
    def __init__(self, parser, data):
        tf.reset_default_graph()

        args = parser.parse_args()
        self.continue_from_epoch = args.continue_from_epoch
        self.experiment_name = args.experiment_title
        self.saved_models_filepath, self.log_path, self.save_image_path = build_experiment_folder(self.experiment_name)
        self.num_gpus = args.num_of_gpus
        self.batch_size = args.batch_size
        gen_depth_per_layer = args.generator_inner_layers
        discr_depth_per_layer = args.discriminator_inner_layers
        self.z_dim = args.z_dim
        self.num_generations = args.num_generations
        self.dropout_rate_value = args.dropout_rate_value
        self.data = data
        self.reverse_channels = False

        generator_layers = [64, 64, 128, 128]
        discriminator_layers = [64, 64, 128, 128]

        gen_inner_layers = [gen_depth_per_layer, gen_depth_per_layer, gen_depth_per_layer, gen_depth_per_layer]
        discr_inner_layers = [discr_depth_per_layer, discr_depth_per_layer, discr_depth_per_layer,
                              discr_depth_per_layer]
        generator_layer_padding = ["SAME", "SAME", "SAME", "SAME"]

        image_height = data.image_height
        image_width = data.image_width
        image_channel = data.image_channel

        self.input_x_i = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, image_height, image_width,
                                                     image_channel], 'inputs-1')
        self.input_x_j = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, image_height, image_width,
                                                     image_channel], 'inputs-2-same-class')

        self.z_input = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], 'z-input')
        self.training_phase = tf.placeholder(tf.bool, name='training-flag')
        self.random_rotate = tf.placeholder(tf.bool, name='rotation-flag')
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout-prob')

        dagan = DAGAN(batch_size=self.batch_size, input_x_i=self.input_x_i, input_x_j=self.input_x_j,
                      dropout_rate=self.dropout_rate, generator_layer_sizes=generator_layers,
                      generator_layer_padding=generator_layer_padding, num_channels=data.image_channel,
                      is_training=self.training_phase, augment=self.random_rotate,
                      discriminator_layer_sizes=discriminator_layers,
                      discr_inner_conv=discr_inner_layers,
                      gen_inner_conv=gen_inner_layers, num_gpus=self.num_gpus, z_dim=self.z_dim, z_inputs=self.z_input)

        self.summary, self.losses, self.graph_ops = dagan.init_train()
        self.same_images = dagan.sample_same_images()

        self.total_train_batches = data.training_data_size / (self.batch_size * self.num_gpus)
        self.total_val_batches = data.validation_data_size / (self.batch_size * self.num_gpus)
        self.total_test_batches = data.testing_data_size / (self.batch_size * self.num_gpus)
        self.total_gen_batches = data.generation_data_size / (self.batch_size * self.num_gpus)
        self.init = tf.global_variables_initializer()
        self.spherical_interpolation = True
        self.tensorboard_update_interval = int(self.total_train_batches/100/self.num_gpus)
        self.total_epochs = 200

        if self.continue_from_epoch == -1:
            save_statistics(self.log_path, ["epoch", "total_d_loss", "total_g_loss", "total_d_val_loss",
                                              "total_g_val_loss"], create=True)

    def run_experiment(self):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(self.init)
            self.writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            self.saver = tf.train.Saver()
            start_from_epoch = 0
            if self.continue_from_epoch!=-1:
                start_from_epoch = self.continue_from_epoch
                checkpoint = "{}/{}_{}.ckpt".format(self.saved_models_filepath, self.experiment_name, self.continue_from_epoch)
                variables_to_restore = []
                for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                    print(var)
                    variables_to_restore.append(var)

                tf.logging.info('Fine-tuning from %s' % checkpoint)

                fine_tune = slim.assign_from_checkpoint_fn(
                    checkpoint,
                    variables_to_restore,
                    ignore_missing_vars=True)
                fine_tune(sess)

            self.iter_done = 0
            self.disc_iter = 5
            self.gen_iter = 1

            if self.spherical_interpolation:
                dim = int(np.sqrt(self.num_generations)*2)
                self.z_2d_vectors = interpolations.create_mine_grid(rows=dim,
                                                                    cols=dim,
                                                                    dim=self.z_dim, space=3, anchors=None,
                                                                    spherical=True, gaussian=True)
                self.z_vectors = interpolations.create_mine_grid(rows=1, cols=self.num_generations, dim=self.z_dim,
                                                                 space=3, anchors=None, spherical=True, gaussian=True)
            else:
                self.z_vectors = np.random.normal(size=(self.num_generations, self.z_dim))
                self.z_2d_vectors = np.random.normal(size=(self.num_generations, self.z_dim))

            with tqdm.tqdm(total=self.total_epochs-start_from_epoch) as pbar_e:
                for e in range(start_from_epoch, self.total_epochs):

                    total_g_loss = 0.
                    total_d_loss = 0.
                    save_path = self.saver.save(sess, "{}/{}_{}.ckpt".format(self.saved_models_filepath,
                                                                             self.experiment_name, e))
                    print("Model saved at", save_path)
                    with tqdm.tqdm(total=self.total_train_batches) as pbar_train:
                        x_train_a_gan_list, x_train_b_gan_same_class_list = self.data.get_train_batch()

                        sample_generator(num_generations=self.num_generations, sess=sess, same_images=self.same_images,
                                         inputs=x_train_a_gan_list,
                                         data=self.data, batch_size=self.batch_size, z_input=self.z_input,
                                         file_name="{}/train_z_variations_{}_{}.png".format(self.save_image_path,
                                                                                            self.experiment_name,
                                                                                            e),
                                         input_a=self.input_x_i, training_phase=self.training_phase,
                                         z_vectors=self.z_vectors, dropout_rate=self.dropout_rate,
                                         dropout_rate_value=self.dropout_rate_value)

                        sample_two_dimensions_generator(sess=sess,
                                                        same_images=self.same_images,
                                                        inputs=x_train_a_gan_list,
                                                        data=self.data, batch_size=self.batch_size, z_input=self.z_input,
                                                        file_name="{}/train_z_spherical_{}_{}".format(self.save_image_path,
                                                                                                        self.experiment_name,
                                                                                                        e),
                                                        input_a=self.input_x_i, training_phase=self.training_phase,
                                                        dropout_rate=self.dropout_rate,
                                                        dropout_rate_value=self.dropout_rate_value,
                                                        z_vectors=self.z_2d_vectors)

                        with tqdm.tqdm(total=self.total_gen_batches) as pbar_samp:
                            for i in range(self.total_gen_batches):
                                x_gen_a = self.data.get_gen_batch()
                                sample_generator(num_generations=self.num_generations, sess=sess, same_images=self.same_images,
                                                 inputs=x_gen_a,
                                                 data=self.data, batch_size=self.batch_size, z_input=self.z_input,
                                                 file_name="{}/test_z_variations_{}_{}_{}.png".format(self.save_image_path,
                                                                                                      self.experiment_name, e, i),
                                                 input_a=self.input_x_i, training_phase=self.training_phase,
                                                 z_vectors=self.z_vectors, dropout_rate=self.dropout_rate,
                                                 dropout_rate_value=self.dropout_rate_value)

                                sample_two_dimensions_generator(sess=sess,
                                                                same_images=self.same_images,
                                                                inputs=x_gen_a,
                                                                data=self.data, batch_size=self.batch_size,
                                                                z_input=self.z_input,
                                                                file_name="{}/val_z_spherical_{}_{}_{}".format(
                                                                    self.save_image_path,
                                                                    self.experiment_name,
                                                                    e, i),
                                                                input_a=self.input_x_i,
                                                                training_phase=self.training_phase,
                                                                dropout_rate=self.dropout_rate,
                                                                dropout_rate_value=self.dropout_rate_value,
                                                                z_vectors=self.z_2d_vectors)

                                pbar_samp.update(1)

                        for i in range(self.total_train_batches):

                            for j in range(self.disc_iter):
                                x_train_a_gan_list, x_train_b_gan_same_class_list = self.data.get_train_batch()
                                _, d_loss_value = sess.run(
                                    [self.graph_ops["d_opt_op"], self.losses["d_losses"]],
                                    feed_dict={self.input_x_i: x_train_a_gan_list,
                                               self.input_x_j: x_train_b_gan_same_class_list,
                                               self.dropout_rate: self.dropout_rate_value,
                                               self.training_phase: True, self.random_rotate: True})
                                total_d_loss += d_loss_value

                            for j in range(self.gen_iter):
                                x_train_a_gan_list, x_train_b_gan_same_class_list = \
                                    self.data.get_train_batch()
                                _, g_loss_value, summaries, = sess.run(
                                    [self.graph_ops["g_opt_op"], self.losses["g_losses"], self.summary],
                                    feed_dict={self.input_x_i: x_train_a_gan_list,
                                               self.input_x_j: x_train_b_gan_same_class_list,
                                               self.dropout_rate: self.dropout_rate_value,
                                               self.training_phase: True, self.random_rotate: True})

                                total_g_loss += g_loss_value

                            if i % (self.tensorboard_update_interval) == 0:
                                self.writer.add_summary(summaries)
                            self.iter_done = self.iter_done + 1
                            iter_out = "d_loss: {}, g_loss: {}".format(d_loss_value, g_loss_value)
                            pbar_train.set_description(iter_out)
                            pbar_train.update(1)

                    total_g_loss /= (self.total_train_batches * self.gen_iter)

                    total_d_loss /= (self.total_train_batches * self.disc_iter)

                    print("Epoch {}: d_loss: {}, wg_loss: {}".format(e, total_d_loss, total_g_loss))

                    total_g_val_loss = 0.
                    total_d_val_loss = 0.

                    with tqdm.tqdm(total=self.total_test_batches) as pbar_val:
                        for i in range(self.total_test_batches):

                            for j in range(self.disc_iter):
                                x_test_a, x_test_b = self.data.get_test_batch()
                                d_loss_value = sess.run(self.losses["d_losses"],
                                                        feed_dict={self.input_x_i: x_test_a,
                                                                   self.input_x_j: x_test_b,
                                                                   self.training_phase: False,
                                                                   self.random_rotate: False,
                                                                   self.dropout_rate:
                                                                   self.dropout_rate_value})

                                total_d_val_loss += d_loss_value

                            for j in range(self.gen_iter):
                                x_test_a, x_test_b = self.data.get_test_batch()
                                g_loss_value = sess.run(self.losses["g_losses"],
                                                        feed_dict={self.input_x_i: x_test_a,
                                                                   self.input_x_j: x_test_b,
                                                                   self.training_phase: False,
                                                                   self.random_rotate: False,
                                                                   self.dropout_rate:
                                                                       self.dropout_rate_value})

                                total_g_val_loss += (g_loss_value)

                            self.iter_done = self.iter_done + 1
                            iter_out = "d_loss: {}, g_loss: {}".format(d_loss_value, g_loss_value)
                            pbar_val.set_description(iter_out)
                            pbar_val.update(1)

                    total_g_val_loss /= (self.total_test_batches * self.gen_iter)
                    total_d_val_loss /= (self.total_test_batches * self.disc_iter)

                    print("Epoch {}: d_val_loss: {}, wg_val_loss: {}".format(e, total_d_val_loss, total_g_val_loss))

                    save_statistics(self.log_path, [e, total_d_loss, total_g_loss, total_d_val_loss, total_g_val_loss])

                    pbar_e.update(1)
