# Re-implementing MIMO

Final project for DD2412 Deep Learning, Advanced Course at KTH Royal Institute of Technology, Stockholm.

# Group members

  - Diogo Pinheiro (https://github.com/DiogorPinheiro)
  - Jakob Lindén (https://github.com/jakobGTO)

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
