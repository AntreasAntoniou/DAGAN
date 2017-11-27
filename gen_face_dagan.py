import argparse
import data as dataset
from generation_builder import ExperimentBuilder

parser = argparse.ArgumentParser(description='Welcome to GAN-Shot-Learning script')
parser.add_argument('--batch_size', nargs="?", type=int, default=32, help='batch_size for experiment')
parser.add_argument('--discriminator_inner_layers', nargs="?", type=int, default=1, help='discr_number_of_conv_per_layer')
parser.add_argument('--generator_inner_layers', nargs="?", type=int, default=1, help='discr_number_of_conv_per_layer')
parser.add_argument('--experiment_title', nargs="?", type=str, default="densenet_generator_fc", help='Experiment name')
parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='continue from checkpoint of epoch')
parser.add_argument('--num_of_gpus', nargs="?", type=int, default=1, help='discr_number_of_conv_per_layer')
parser.add_argument('--z_dim', nargs="?", type=int, default=100, help='The dimensionality of the z input')
parser.add_argument('--dropout_rate_value', type=float, default=0.5, help='dropout_rate_value')
parser.add_argument('--num_generations', nargs="?", type=int, default=64, help='num_generations')


args = parser.parse_args()
batch_size = args.batch_size
num_gpus = args.num_of_gpus

data = dataset.VGGFaceDAGANDataset(batch_size=batch_size, last_training_class_index=1600, reverse_channels=True,
                                   num_of_gpus=num_gpus, gen_batches=10)

experiment = ExperimentBuilder(parser, data=data)
experiment.run_experiment()
