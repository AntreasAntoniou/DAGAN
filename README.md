# DAGAN
Implementation of Data Augmentation Generative Adversarial Networks

## Introduction

This is an implementation of the Data Augmentation Generative Adversarial Networks as described in paper https://arxiv.org/abs/1711.04340. The implementation provides data providers, model builders, trainers and data generators for the Omniglot and VGG-Face datasets.

## Installation

To use the DAGAN repository you must first install the project dependencies. This can be conveniently done using:

```pip install -r requirements.txt```

## Datasets

To download the datasets please go to https://drive.google.com/drive/folders/15x2C11OrNeKLMzBDHrv8NPOwyre6H3O5?usp=sharing.

Then download Omniglot or VGG-Face as you see fit for your purposes.

## Training a DAGAN

To train a DAGAN first you must make sure you get the datasets. Once that is done one can train a DAGAN using:

```
python train_omniglot_dagan.py --batch_size 32 --generator_inner_layers 3 --discriminator_inner_layers 5 --num_generations 64 --experiment_title omniglot_dagan_experiment_default --num_of_gpus 1 --z_dim 100 --dropout_rate_value 0.5
```

Where generator and discriminator inner layers represent the number of inner layers per MultiLayer in generator and discriminator respectively. Number of generations refers to how many generated samples should be generated for the spherical interpolations at the end of each epoch.

## Multi-GPU Usage

Our implementation includes multi gpu training, simply pass --num_of_gpus 4 to the script to train on 4 GPUs (only works for multi gpu machines not distributed machines).

## Defining a new task for the DAGAN

If you want to train your own DAGAN on a new dataset you need to do the following:

1. Go in data.py and define a new data provider that inherits from either DAGANDataset or DAGANImblancedDataset. The first one is used when a dataset is balanced i.e. every class has the same amount of samples and the latter when this is not the case.

An example for a balanced dataset is:

```class OmniglotDAGANDataset(DAGANDataset):
    def __init__(self, batch_size, gan_training_index, reverse_channels, num_of_gpus, gen_batches):
        super(OmniglotDAGANDataset, self).__init__(batch_size, gan_training_index, reverse_channels, num_of_gpus,
                                                   gen_batches)

    def load_dataset(self, gan_training_index):

        self.x = np.load("datasets/omniglot_data.npy")
        x_train, x_test, x_val = self.x[:1200], self.x[1200:1600], self.x[1600:]
        x_train = x_train[:gan_training_index]

        return x_train, x_test, x_val
 ```
 
 Basically all you need to do is define the load_dataset function. This function should load your dataset in the form [num_classes, num_samples, im_height, im_width, im_channels]. Then you need to choose which classes go to each of your training, validation and testing sets. Once that is done just return them or if you'd like to have control over how much of your training set is actually used for training you can use the gan_training_index to define where the training set will stop for the current experiment (This allows one to easily change the amount of data used and observe the differences in performance among other things).
 
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

In this case we expect the dataset to have the form [num_classes, num_samples, im_height, im_width, im_channels] but the num samples should be different for each class therefore when the shape is checked it should only return [num_classes,].

## To Generate Data

The model training automatically uses unseen data to produce some generations at the end of each epoch, however once you have trained a model satisfactorily you can generate samples for the whole of the validation set using the following command:

```
python gen_omniglot_dagan.py -batch_size 32 --generator_inner_layers 3 --discriminator_inner_layers 5 --num_generations 64 --experiment_title omniglot_dagan_experiment_default --num_of_gpus 1 --z_dim 100 --dropout_rate_value 0.5 --continue_from_epoch 38
```
All the arguments must match the trained network's arguments and furthermore the continue_from_epoch argument must have the epoch of the best model we want to generate from.

## Additional generated data not shown in the paper

For further generated data please visit https://drive.google.com/drive/folders/1IqdhiQzxHysSSnfSrGA9_jKTWzp9gl0k?usp=sharing.

## Acknowledgements

Special thanks to the CDT in Data Science at the University of Edinburgh for providing the funding and resources for this project.
Furthermore, special thanks to my colleague James Owers for reviewing this code and providing improvements and suggestions.
