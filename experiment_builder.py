import utils.interpolations as interpolations
import numpy as np
import tqdm
from utils.storage import save_statistics, build_experiment_folder
from tensorflow.contrib import slim

from dagan_networks_wgan import *
from utils.sampling import sample_generator, sample_two_dimensions_generator


class ExperimentBuilder(object):
    def __init__(self, args, data):
        tf.reset_default_graph()

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
                      gen_inner_conv=gen_inner_layers, num_gpus=self.num_gpus, z_dim=self.z_dim, z_inputs=self.z_input,
                      use_wide_connections=args.use_wide_connections)

        self.summary, self.losses, self.graph_ops = dagan.init_train()
        self.same_images = dagan.sample_same_images()

        self.total_train_batches = int(data.training_data_size / (self.batch_size * self.num_gpus))

        self.total_gen_batches = int(data.generation_data_size / (self.batch_size * self.num_gpus))

        self.init = tf.global_variables_initializer()
        self.spherical_interpolation = True
        self.tensorboard_update_interval = int(self.total_train_batches/100/self.num_gpus)
        self.total_epochs = 200

        if self.continue_from_epoch == -1:
            save_statistics(self.log_path, ['epoch', 'total_d_train_loss_mean', 'total_d_val_loss_mean',
                                            'total_d_train_loss_std', 'total_d_val_loss_std',
                                            'total_g_train_loss_mean', 'total_g_val_loss_mean',
                                            'total_g_train_loss_std', 'total_g_val_loss_std'], create=True)

    def run_experiment(self):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(self.init)
            self.train_writer = tf.summary.FileWriter("{}/train_logs/".format(self.log_path),
                                                      graph=tf.get_default_graph())
            self.validation_writer = tf.summary.FileWriter("{}/validation_logs/".format(self.log_path),
                                                           graph=tf.get_default_graph())
            self.train_saver = tf.train.Saver()
            self.val_saver = tf.train.Saver()

            start_from_epoch = 0
            if self.continue_from_epoch!=-1:
                start_from_epoch = self.continue_from_epoch
                checkpoint = "{}train_saved_model_{}_{}.ckpt".format(self.saved_models_filepath, self.experiment_name, self.continue_from_epoch)
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
            best_d_val_loss = np.inf

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

                    train_g_loss = []
                    val_g_loss = []
                    train_d_loss = []
                    val_d_loss = []

                    with tqdm.tqdm(total=self.total_train_batches) as pbar_train:
                        for iter in range(self.total_train_batches):

                            cur_sample = 0

                            for n in range(self.disc_iter):
                                x_train_i, x_train_j = self.data.get_train_batch()
                                x_val_i, x_val_j = self.data.get_val_batch()

                                _, d_train_loss_value = sess.run(
                                    [self.graph_ops["d_opt_op"], self.losses["d_losses"]],
                                    feed_dict={self.input_x_i: x_train_i,
                                               self.input_x_j: x_train_j,
                                               self.dropout_rate: self.dropout_rate_value,
                                               self.training_phase: True, self.random_rotate: True})

                                d_val_loss_value = sess.run(
                                    self.losses["d_losses"],
                                    feed_dict={self.input_x_i: x_val_i,
                                               self.input_x_j: x_val_j,
                                               self.dropout_rate: self.dropout_rate_value,
                                               self.training_phase: False, self.random_rotate: False})

                                cur_sample += 1
                                train_d_loss.append(d_train_loss_value)
                                val_d_loss.append(d_val_loss_value)

                            for n in range(self.gen_iter):
                                x_train_i, x_train_j = self.data.get_train_batch()
                                x_val_i, x_val_j = self.data.get_val_batch()
                                _, g_train_loss_value, train_summaries = sess.run(
                                    [self.graph_ops["g_opt_op"], self.losses["g_losses"],
                                     self.summary],
                                    feed_dict={self.input_x_i: x_train_i,
                                               self.input_x_j: x_train_j,
                                               self.dropout_rate: self.dropout_rate_value,
                                               self.training_phase: True, self.random_rotate: True})

                                g_val_loss_value, val_summaries = sess.run(
                                    [self.losses["g_losses"], self.summary],
                                    feed_dict={self.input_x_i: x_val_i,
                                               self.input_x_j: x_val_j,
                                               self.dropout_rate: self.dropout_rate_value,
                                               self.training_phase: False, self.random_rotate: False})

                                cur_sample += 1
                                train_g_loss.append(g_train_loss_value)
                                val_g_loss.append(g_val_loss_value)

                                if iter % (self.tensorboard_update_interval) == 0:
                                    self.train_writer.add_summary(train_summaries, global_step=self.iter_done)
                                    self.validation_writer.add_summary(val_summaries, global_step=self.iter_done)


                            self.iter_done = self.iter_done + 1
                            iter_out = "{}_train_d_loss: {}, train_g_loss: {}, " \
                                       "val_d_loss: {}, val_g_loss: {}".format(self.iter_done,
                                                                               d_train_loss_value, g_train_loss_value,
                                                                               d_val_loss_value,
                                                                               g_val_loss_value)
                            pbar_train.set_description(iter_out)
                            pbar_train.update(1)

                    total_d_train_loss_mean = np.mean(train_d_loss)
                    total_d_train_loss_std = np.std(train_d_loss)
                    total_g_train_loss_mean = np.mean(train_g_loss)
                    total_g_train_loss_std = np.std(train_g_loss)

                    print(
                        "Epoch {}: d_train_loss_mean: {}, d_train_loss_std: {},"
                                  "g_train_loss_mean: {}, g_train_loss_std: {}"
                        .format(e, total_d_train_loss_mean,
                                total_d_train_loss_std,
                                total_g_train_loss_mean,
                                total_g_train_loss_std))

                    total_d_val_loss_mean = np.mean(val_d_loss)
                    total_d_val_loss_std = np.std(val_d_loss)
                    total_g_val_loss_mean = np.mean(val_g_loss)
                    total_g_val_loss_std = np.std(val_g_loss)

                    print(
                        "Epoch {}: d_val_loss_mean: {}, d_val_loss_std: {},"
                        "g_val_loss_mean: {}, g_val_loss_std: {}, "
                            .format(e, total_d_val_loss_mean,
                                    total_d_val_loss_std,
                                    total_g_val_loss_mean,
                                    total_g_val_loss_std))



                    sample_generator(num_generations=self.num_generations, sess=sess, same_images=self.same_images,
                                     inputs=x_train_i,
                                     data=self.data, batch_size=self.batch_size, z_input=self.z_input,
                                     file_name="{}/train_z_variations_{}_{}.png".format(self.save_image_path,
                                                                                        self.experiment_name,
                                                                                        e),
                                     input_a=self.input_x_i, training_phase=self.training_phase,
                                     z_vectors=self.z_vectors, dropout_rate=self.dropout_rate,
                                     dropout_rate_value=self.dropout_rate_value)

                    sample_two_dimensions_generator(sess=sess,
                                                    same_images=self.same_images,
                                                    inputs=x_train_i,
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
                            sample_generator(num_generations=self.num_generations, sess=sess,
                                             same_images=self.same_images,
                                             inputs=x_gen_a,
                                             data=self.data, batch_size=self.batch_size, z_input=self.z_input,
                                             file_name="{}/test_z_variations_{}_{}_{}.png".format(self.save_image_path,
                                                                                                  self.experiment_name,
                                                                                                  e, i),
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

                    train_save_path = self.train_saver.save(sess, "{}/train_saved_model_{}_{}.ckpt".format(
                        self.saved_models_filepath,
                        self.experiment_name, e))

                    if total_d_val_loss_mean<best_d_val_loss:
                        best_d_val_loss = total_d_val_loss_mean
                        val_save_path = self.train_saver.save(sess, "{}/val_saved_model_{}_{}.ckpt".format(
                            self.saved_models_filepath,
                            self.experiment_name, e))
                        print("Saved current best val model at", val_save_path)

                    save_statistics(self.log_path, [e, total_d_train_loss_mean, total_d_val_loss_mean,
                                                total_d_train_loss_std, total_d_val_loss_std,
                                                total_g_train_loss_mean, total_g_val_loss_mean,
                                                total_g_train_loss_std, total_g_val_loss_std])

                    pbar_e.update(1)

