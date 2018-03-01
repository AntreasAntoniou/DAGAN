import csv

def save_statistics(log_path, line_to_add, log_name="experiment_log.csv", create=False):
    if create:
        with open("{}/{}".format(log_path, log_name), 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(line_to_add)
    else:
        with open("{}/{}".format(log_path, log_name), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(line_to_add)

def load_statistics(log_path, log_name="experiment_log"):
    data_dict = dict()
    with open("{}/{}".format(log_path, log_name), 'r') as f:
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
    saved_models_filepath = "{}/{}".format(experiment_name.replace("_", "/"), "saved_models/")
    logs_filepath = "{}/{}".format(experiment_name.replace("_", "/"), "logs/")
    samples_filepath = "{}/{}".format(experiment_name.replace("_", "/"), "visual_outputs/")

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
