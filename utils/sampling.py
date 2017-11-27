import scipy.misc
import numpy as np

def unstack(np_array):
    new_list = []
    for i in range(np_array.shape[0]):
        temp_list = np_array[i]
        new_list.append(temp_list)
    return new_list

def sample_generator(num_generations, sess, same_images, inputs, dropout_rate, dropout_rate_value, data, batch_size,
                     file_name, input_a, training_phase, z_input, z_vectors):

    input_images, generated = sess.run(same_images, feed_dict={input_a: inputs, dropout_rate: dropout_rate_value,
                                                                  training_phase: False,
                                                                  z_input: batch_size*[z_vectors[0]]})
    input_images_list = np.zeros(shape=(batch_size, num_generations, input_images.shape[-3], input_images.shape[-2],
                                        input_images.shape[-1]))
    generated_list = np.zeros(shape=(batch_size, num_generations, generated.shape[-3], generated.shape[-2],
                                     generated.shape[-1]))
    height = generated.shape[-3]
    for i in range(num_generations):
        input_images, generated = sess.run(same_images, feed_dict={z_input: batch_size*[z_vectors[i]],
                                                                      input_a: inputs,
                                                                      training_phase: False, dropout_rate:
                                                                      dropout_rate_value})
        input_images_list[:, i] = input_images
        generated_list[:, i] = generated


    input_images, generated = data.reconstruct_original(input_images_list), data.reconstruct_original(generated_list)

    input_images = unstack(input_images)
    input_images = np.concatenate((input_images), axis=1)
    input_images = unstack(input_images)
    input_images = np.concatenate((input_images), axis=1)
    line = np.zeros(shape=(batch_size, 1, generated.shape[-3], generated.shape[-2],
                                     generated.shape[-1]))

    generated = unstack(generated)
    generated = np.concatenate((generated), axis=1)
    generated = unstack(generated)
    generated = np.concatenate((generated), axis=1)

    image = np.concatenate((input_images, generated), axis=1)
    image = np.squeeze(image)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image * 255
    image = image[:, (num_generations-1)*height:]
    scipy.misc.imsave(file_name, image)

def sample_two_dimensions_generator(sess, same_images, inputs,
                                    dropout_rate, dropout_rate_value, data,
                                    batch_size, file_name, input_a,
                                    training_phase, z_input, z_vectors):
    num_generations = z_vectors.shape[0]
    row_num_generations = int(np.sqrt(num_generations))
    column_num_generations = int(np.sqrt(num_generations))

    input_images, generated = sess.run(same_images, feed_dict={input_a: inputs, dropout_rate: dropout_rate_value,
                                                                  training_phase: False,
                                                                  z_input: batch_size*[z_vectors[0]]})

    input_images_list = np.zeros(shape=(batch_size, num_generations, input_images.shape[-3], input_images.shape[-2],
                                        input_images.shape[-1]))
    generated_list = np.zeros(shape=(batch_size, num_generations, generated.shape[-3], generated.shape[-2],
                                     generated.shape[-1]))
    height = generated.shape[-3]

    for i in range(num_generations):
        input_images, generated = sess.run(same_images, feed_dict={z_input: batch_size*[z_vectors[i]],
                                                                      input_a: inputs,
                                                                      training_phase: False, dropout_rate:
                                                                      dropout_rate_value})
        input_images_list[:, i] = input_images
        generated_list[:, i] = generated


    input_images, generated = data.reconstruct_original(input_images_list), data.reconstruct_original(generated_list)
    im_size = generated.shape

    input_images = unstack(input_images)
    input_images = np.concatenate((input_images), axis=1)
    input_images = unstack(input_images)
    input_images = np.concatenate((input_images), axis=1)
    line = np.zeros(shape=(batch_size, 1, generated.shape[-3], generated.shape[-2],
                                     generated.shape[-1]))

    generated = unstack(generated)
    generated = np.concatenate((generated), axis=1)
    generated = unstack(generated)
    generated = np.concatenate((generated), axis=1)

    image = np.concatenate((input_images, generated), axis=1)
    im_dimension = im_size[3]
    image = np.squeeze(image)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image * 255
    full_image = image[:, (num_generations-1)*height:]


    for i in range(batch_size):
        image = full_image[i*im_dimension:(i+1)*im_dimension]
        seed_image = image[0:im_dimension, 0:im_dimension]
        gen_images = image[0:im_dimension, 2*im_dimension:]
        image = np.concatenate((seed_image, gen_images), axis=1)

        properly_positioned_image = []
        for j in range(row_num_generations):
            start = im_dimension*j*row_num_generations
            stop = im_dimension*(j+1)*row_num_generations


            row_image = image[:, start:stop]

            properly_positioned_image.append(row_image)

        positioned_image = np.concatenate(properly_positioned_image, axis=0)

        scipy.misc.imsave("{}_{}.png".format(file_name, i), positioned_image)


