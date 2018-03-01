# DAGAN
Implementation of DAGAN: Data Augmentation Generative Adversarial Networks

## Introduction

This is an implementation of DAGAN as described in https://arxiv.org/abs/1711.04340. The implementation provides data loaders, model builders, model trainers, and synthetic data generators for the Omniglot and VGG-Face datasets.

## Installation

To use the DAGAN repository you must first install the project dependencies. This can be done by install miniconda3 from <a href="https://conda.io/miniconda.html">here</a> 
 with python 3 and running:

```pip install -r requirements.txt```

## Datasets

The Omniglot and VGG-Face datasets can be obtained in numpy format <a href="https://drive.google.com/drive/folders/15x2C11OrNeKLMzBDHrv8NPOwyre6H3O5?usp=sharing" target="_blank">here</a>. They should then be placed in the `datasets` folder.

## Training a DAGAN

After the datasets are downloaded and the dependencies are installed, a DAGAN can be trained by running:

```
python train_omniglot_dagan.py --batch_size 32 --generator_inner_layers 3 --discriminator_inner_layers 5 --num_generations 64 --experiment_title omniglot_dagan_experiment_default --num_of_gpus 1 --z_dim 100 --dropout_rate_value 0.5
```

Here, `generator_inner_layers` and `discriminator_inner_layers` refer to the number of inner layers per MultiLayer in the generator and discriminator respectively. `num_generations` refers to the number of samples generated for use in the spherical interpolations at the end of each epoch.

## Multi-GPU Usage

Our implementation supports multi-GPU training. Simply pass `--num_of_gpus <x>` to the script to train on  x GPUs (note that this only works if the GPUs are on the same machine).

## Defining a new task for the DAGAN

If you want to train your own DAGAN on a new dataset you need to do the following:

1. Edit data.py and define a new data loader class that inherits from either DAGANDataset or DAGANImblancedDataset. The first class is used when a dataset is balanced (i.e. every class has the same number of samples), the latter is for when this is not the case.

An example class for a balanced dataset is:

```
class OmniglotDAGANDataset(DAGANDataset):
    def __init__(self, batch_size, gan_training_index, reverse_channels, num_of_gpus, gen_batches):
        super(OmniglotDAGANDataset, self).__init__(batch_size, gan_training_index, reverse_channels, num_of_gpus,
                                                   gen_batches)

    def load_dataset(self, gan_training_index):

        self.x = np.load("datasets/omniglot_data.npy")
        x_train, x_test, x_val = self.x[:1200], self.x[1200:1600], self.x[1600:]
        x_train = x_train[:gan_training_index]

        return x_train, x_test, x_val
 ```
 
 An example for an imbalanced dataset is:
 
 ```
 class OmniglotImbalancedDAGANDataset(DAGANImbalancedDataset):
    def __init__(self, batch_size, gan_training_index, reverse_channels, num_of_gpus, gen_batches):
        super(OmniglotImbalancedDAGANDataset, self).__init__(batch_size, gan_training_index, reverse_channels,
                                                             num_of_gpus, gen_batches)

    def load_dataset(self, gan_training_index):

        x = np.load("datasets/omniglot_data.npy")
        x_temp = []
        for i in range(x.shape[0]):
            choose_samples = np.random.choice([i for i in range(1, 15)])
            x_temp.append(x[i, :choose_samples])
        self.x = np.array(x_temp)
        x_train, x_test, x_val = self.x[:1200], self.x[1200:1600], self.x[1600:]
        x_train = x_train[:gan_training_index]

        return x_train, x_test, x_val
 ```

In short, you need to define your own load_dataset function. This function should load your dataset in the form [num_classes, num_samples, im_height, im_width, im_channels]. Make sure your data values lie within the 0.0 to 1.0 range otherwise the system will fail to model them. Then you need to choose which classes go to each of your training, validation and test sets.

2. Once your data loader is ready, use a template such as train_omniglot_dagan.py and change the data loader that is being passed. This should be sufficient to run experiments on any new image dataset.

## To Generate Data

The model training automatically uses unseen data to produce generations at the end of each epoch. However, once you have trained a model to satisfication you can generate samples for the whole of the validation set using the following command:

```
python gen_omniglot_dagan.py -batch_size 32 --generator_inner_layers 3 --discriminator_inner_layers 5 --num_generations 64 --experiment_title omniglot_dagan_experiment_default --num_of_gpus 1 --z_dim 100 --dropout_rate_value 0.5 --continue_from_epoch 38
```
All the arguments must match the trained network's arguments and the `continue_from_epoch` argument must correspond to the epoch the trained model was at.

## Additional generated data not shown in the paper

For further generated data please visit 
<a href="https://drive.google.com/drive/folders/1IqdhiQzxHysSSnfSrGA9_jKTWzp9gl0k?usp=sharing" target="_blank">my Google Drive folder</a>.

## Acknowledgements

Special thanks to the CDT in Data Science at the University of Edinburgh for providing the funding and resources for this project.
Furthermore, special thanks to my colleagues James Owers, Todor Davchev, Elliot Crowley, and Gavin Gray for reviewing this code and providing improvements and suggestions.

Furthermore, the interpolations used in this project are a result of the <a href="https://arxiv.org/abs/1609.04468" target="_blank">Sampling Generative Networks paper</a> by Tom White. 
The code itself was found at https://github.com/dribnet/plat.
