The only difference between MIMO and a standard NN is the input and output. Instead of taking a single datapoint we take M datapoints as input and M nodes as output.

The paper benchmarks on ResNet28-10 on CIFAR10 and CIFAR 100 and ResNet-50 for ImageNet. 

In ResNet-X-Y, X defines the depth of the network and Y is the width multiplier, i.e. what scalar we would multiply the filtersize with when widening the network.

# Gameplan

1. Build two mimo classes, one for resnet-28-10 and one for resnet-50
2. Implement function to build resnet blocks that will be used in creating the model  
3. In resnet-28-10 use normal resnet blocks, in resnet-50 use bottleneck resnet blocks.
4. Implement function to build the model with resnet architecture
5. Implement train step and fit functions