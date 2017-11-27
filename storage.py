from PIL import Image
import numpy as np
import os
import csv

def save_experiment(experiment_name, train_dict, val_dict):
    import pickle
    order_dicts = ["train", "valid"]
    for i, entry in enumerate([train_dict, val_dict]):
        with open('{}_{}.pkl'.format(order_dicts[i], experiment_name), 'wb') as fp:
            pickle.dump(entry, fp, pickle.HIGHEST_PROTOCOL)

def save_images_3_channels(batch_in, batch_gen, name, rows, columns, batch_r_gen=None):
    if batch_r_gen is None:
        images_in = []
        images_out = []
        for image_in, image_out in zip(batch_in, batch_gen):
            images_in.append(Image.fromarray(np.uint8(image_in)))
            images_out.append(Image.fromarray(np.uint8(image_out)))

        widths, heights = zip(*(i.size for i in images_in))
        if columns==None:
            images_per_row = int(len(batch_in) / rows)
        else:
            images_per_row = columns

        total_width = widths[0] * (images_per_row * 2 + 1)
        max_height = heights[0] * columns
        new_im = Image.new('RGB', (total_width, max_height))

        y_offset = 0

        for j in range(rows):
            x_offset = 0
            for i in range(images_per_row * 2 + 1):
                if i < images_per_row:
                    new_im.paste(images_in[images_per_row * j + i], (x_offset, y_offset))
                elif i > images_per_row:
                    new_im.paste(images_out[images_per_row * j + i - images_per_row - 1], (x_offset, y_offset))

                x_offset += images_in[0].size[0]
            y_offset += images_in[0].size[1]

    else:
        images_in = []
        images_out = []
        images_r_out = []

        for image_in, image_out, image_r_out in zip(batch_in, batch_gen, batch_r_gen):
            images_in.append(Image.fromarray(np.uint8(image_in)))
            images_out.append(Image.fromarray(np.uint8(image_out)))
            images_r_out.append(Image.fromarray(np.uint8(image_r_out)))


        widths, heights = zip(*(i.size for i in images_in))
        if columns == None:
            images_per_row = int(len(batch_in) / rows)
        else:
            images_per_row = columns

        total_width = widths[0] * (images_per_row * 3 + 2)
        max_height = heights[0] * columns

        new_im = Image.new('RGB', (total_width, max_height))

        y_offset = 0

        for j in range(rows):
            x_offset = 0
            for i in range(images_per_row * 3 + 2):
                if i < images_per_row:
                    new_im.paste(images_in[images_per_row * j + i], (x_offset, y_offset))

                elif i > images_per_row and i<2*images_per_row+1:
                    new_im.paste(images_out[images_per_row * j + i - images_per_row - 1], (x_offset, y_offset))
                elif i > 2*images_per_row + 1:
                    new_im.paste(images_r_out[images_per_row * j + i - (2*(images_per_row - 1))-4], (x_offset, y_offset))

                x_offset += images_in[0].size[0]
            y_offset += images_in[0].size[1]
    directory_prefix = "/".join(name.split("/")[:-1])
    if not os.path.exists(directory_prefix):
        os.makedirs(directory_prefix)
    new_im.save(name)

def save_images_1_channel(batch_in, batch_gen, name, columns, rows, batch_r_gen=None):
    if batch_r_gen == None:
        images_in = []
        images_out = []

        for image_in, image_out in zip(batch_in, batch_gen):
            images_in.append(Image.fromarray(np.uint8(image_in[:, :, 0]), mode="L"))
            images_out.append(Image.fromarray(np.uint8(image_out[:, :, 0]), mode="L"))

        widths, heights = zip(*(i.size for i in images_in))

        if columns==None:
            images_per_row = int(len(batch_in) / rows)
        else:
            images_per_row = columns
        total_width = widths[0] * (images_per_row * 2 + 1)
        max_height = heights[0] * columns
        empty_image = np.zeros((batch_in.shape[1], batch_in.shape[2], batch_in.shape[3]))
        empty_image = Image.fromarray(np.uint8(empty_image[:, :, 0]))
        new_im = Image.new('L', (total_width, max_height), color=255)

        y_offset = 0

        for j in range(rows):
            x_offset = 0
            for i in range(images_per_row * 2 + 1):
                if i < images_per_row:
                    new_im.paste(images_in[images_per_row * j + i], (x_offset, y_offset))

                elif i == images_per_row:
                    new_im.paste(empty_image, (x_offset, y_offset))

                elif i > images_per_row:
                    new_im.paste(images_out[images_per_row * j + i - images_per_row - 1], (x_offset, y_offset))

                x_offset += images_in[0].size[0]
            y_offset += images_in[0].size[1]
    else:
        images_in = []
        images_out = []
        images_r_out = []

        for image_in, image_out, image_r_out in zip(batch_in, batch_gen, batch_r_gen):
            images_in.append(Image.fromarray(np.uint8(image_in[:, :, 0]), mode="L"))
            images_out.append(Image.fromarray(np.uint8(image_out[:, :, 0]), mode="L"))
            images_r_out.append(Image.fromarray(np.uint8(image_r_out[:, :, 0]), mode="L"))

        widths, heights = zip(*(i.size for i in images_in))

        if columns == None:
            images_per_row = int(len(batch_in) / rows)
        else:
            images_per_row = columns

        total_width = widths[0] * (images_per_row * 3 + 2)
        max_height = heights[0] * columns
        empty_image = np.zeros((batch_in.shape[1], batch_in.shape[2], batch_in.shape[3]))
        empty_image = Image.fromarray(np.uint8(empty_image[:, :, 0]))
        new_im = Image.new('L', (total_width, max_height), color=255)

        y_offset = 0
        for j in range(rows):
            x_offset = 0
            for i in range(images_per_row * 3 + 2):
                if i < images_per_row:
                    new_im.paste(images_in[images_per_row * j + i], (x_offset, y_offset))
                elif i == images_per_row:
                    new_im.paste(empty_image, (x_offset, y_offset))
                elif i > images_per_row and i<2*images_per_row+1:
                    new_im.paste(images_out[images_per_row * j + i - images_per_row - 1], (x_offset, y_offset))
                elif i == 2*images_per_row + 1:
                    new_im.paste(empty_image, (x_offset, y_offset))
                elif i > 2*images_per_row + 1:
                    new_im.paste(images_r_out[images_per_row * j + i - (2*(images_per_row - 1))-4], (x_offset, y_offset))

                x_offset += images_in[0].size[0]
            y_offset += images_in[0].size[1]

    new_im.save(name)

def save_images(batch_in, batch_gen, name, batch_r_gen=None, rows=10, columns=5, channels=3, reverse_colours=False):

    batch_in = (batch_in - np.min(batch_in)) / (np.max(batch_in) - np.min(batch_in))
    batch_in *= 255
    batch_gen = (batch_gen - np.min(batch_gen)) / (np.max(batch_gen) - np.min(batch_gen))
    batch_gen *= 255
    if batch_r_gen is not None:
        batch_r_gen = (batch_r_gen - np.min(batch_r_gen)) / (np.max(batch_r_gen) - np.min(batch_r_gen))
        batch_r_gen *= 255
    if reverse_colours:
        batch_in_temp = np.zeros((batch_in.shape))
        batch_gen_temp = np.ones((batch_gen.shape))
        if batch_r_gen is not None:
            batch_r_gen_temp = np.ones((batch_r_gen.shape))
        for i in range(3):
            batch_in_temp[:, :, :, i] = batch_in[:, :, :, 2-i]
            batch_gen_temp[:, :, :, i] = batch_gen[:, :, :, 2-i]
            if batch_r_gen is not None:
                batch_r_gen_temp[:, :, :, i] = batch_r_gen[:, :, :, 2-i]
        batch_in = batch_in_temp
        batch_gen = batch_gen_temp
        if batch_r_gen is not None:
            batch_r_gen = batch_r_gen_temp

    if channels==1:
        return save_images_1_channel(batch_in, batch_gen, name, rows, columns, batch_r_gen=batch_r_gen)
    elif channels==3:
        return save_images_3_channels(batch_in, batch_gen, name, rows, columns, batch_r_gen=batch_r_gen)

def save_statistics(experiment_name, line_to_add, create=False):
    if create:
        with open("{}.csv".format(experiment_name), 'wb+') as f:
            writer = csv.writer(f)
            writer.writerow(line_to_add)
    else:
        with open("{}.csv".format(experiment_name), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(line_to_add)

def load_statistics(experiment_name):
    data_dict = dict()
    with open("{}.csv".format(experiment_name), 'r') as f:
        lines = f.readlines()
        data_labels = lines[0].replace("\n","").split(",")
        del lines[0]

        for label in data_labels:
            data_dict[label] = []

        for line in lines:
            data = line.replace("\n","").split(",")
            for key, item in zip(data_labels, data):
                data_dict[key].append(item)
    return data_dict

def build_experiment_folder(experiment_name):
    saved_models_filepath = "{}/{}".format(experiment_name.replace("_", "/"), "saved_models")
    logs_filepath = "{}/{}".format(experiment_name.replace("_", "/"), "logs")
    samples_filepath = "{}/{}".format(experiment_name.replace("_", "/"), "visual_outputs")

    import os
    if not os.path.exists(experiment_name.replace("_", "/")):
        os.makedirs(experiment_name.replace("_", "/"))
    if not os.path.exists(logs_filepath):
        os.makedirs(logs_filepath)
    if not os.path.exists(samples_filepath):
        os.makedirs(samples_filepath)
    if not os.path.exists(saved_models_filepath):
        os.makedirs(saved_models_filepath)

    return saved_models_filepath, logs_filepath, samples_filepath
