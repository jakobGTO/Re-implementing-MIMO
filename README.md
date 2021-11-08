The only difference between MIMO and a standard NN is the input and output. Instead of taking a single datapoint we take M datapoints as input and M nodes as output.

The paper benchmarks on ResNet28-10 on CIFAR10 and CIFAR 100 and ResNet-50 for ImageNet. 

In ResNet-X-Y, X defines the depth of the network and Y is the width multiplier, i.e. what scalar we would multiply the filtersize with when widening the network.

# Gameplan

1. Build two mimo classes, one for resnet-28-10 and one for resnet-50
2. Implement function to build resnet blocks that will be used in creating the model  
3. In resnet-28-10 use normal resnet blocks, in resnet-50 use bottleneck resnet blocks.
4. Implement function to build the model with resnet architecture
5. Implement train step and fit functions

# Dependencies
1. Tensorflow
2. Numpy
3. Robustness-metrics (source: https://github.com/google-research/robustness_metrics)
4. Matplotlib
5. Uncertainty Baselines (source: https://github.com/google/uncertainty-baselines)

# Running Uncertainty-Baseline comparison tests
1. Clone repository from https://github.com/google/uncertainty-baselines.git
2. Place script "run-tests" in the root directory of the cloned repository
3. Run command "chmod u+x run-tests" in that same directory
4. Run tests in root directory using 
   ```
   ./run-tests --dataset cifar100 --model batchensemble --gpu True --download True --cores 4 --requirements False --epochs 10
   ```
   The following parameters are currently provided:
        - dataset : "cifar100" or "cifar10"
        - model : "all" (will run all pre-defined models) or write a specific one (e.g. "batchensemble" or "dropout")
        - gpu : Boolean that represents whether the process will be ran using gpu or not
        - download: If set to True it will download the required dataset to a temporary directory
        - cores : Number of cores to be user
        - requirements: If set to True it will first and foremost install all necessary dependencies
        - epochs : Number of epochs to be performed


# Results
Epoch: 0 | NLL: 5.7946253 | Time per epoch: 199.031635761261
Epoch: 1 | NLL: 5.047467 | Time per epoch: 189.9454185962677
Epoch: 2 | NLL: 4.688809 | Time per epoch: 199.76514172554016

