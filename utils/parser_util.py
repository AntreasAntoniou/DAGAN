import argparse


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def get_args():
    parser = argparse.ArgumentParser(description='Welcome to GAN-Shot-Learning script')
    parser.add_argument('--batch_size', nargs="?", type=int, default=32, help='batch_size for experiment')
    parser.add_argument('--discriminator_inner_layers', nargs="?", type=int, default=1,
                        help='Number of inner layers per multi layer in the discriminator')
    parser.add_argument('--generator_inner_layers', nargs="?", type=int, default=1,
                        help='Number of inner layers per multi layer in the generator')
    parser.add_argument('--experiment_title', nargs="?", type=str, default="omniglot_dagan_experiment",
                        help='Experiment name')
    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1,
                        help='continue from checkpoint of epoch')
    parser.add_argument('--num_of_gpus', nargs="?", type=int, default=1, help='Number of GPUs to use for training')
    parser.add_argument('--z_dim', nargs="?", type=int, default=100, help='The dimensionality of the z input')
    parser.add_argument('--dropout_rate_value', type=float, default=0.5,
                        help='A dropout rate placeholder or a scalar to use throughout the network')
    parser.add_argument('--num_generations', nargs="?", type=int, default=64,
                        help='The number of samples generated for use in the spherical interpolations at the end of '
                             'each epoch')
    parser.add_argument('--use_wide_connections', nargs="?", type=str, default="False",
                        help='Whether to use wide connections in discriminator')
    args = parser.parse_args()
    batch_size = args.batch_size
    num_gpus = args.num_of_gpus

    args_dict = vars(args)
    for key in list(args_dict.keys()):
        print(key, args_dict[key])

        if args_dict[key] == "True":
            args_dict[key] = True
        elif args_dict[key] == "False":
            args_dict[key] = False
    args = Bunch(args_dict)

    return batch_size, num_gpus, args