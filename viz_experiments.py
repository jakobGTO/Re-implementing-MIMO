import numpy as np
import matplotlib.pyplot as plt



def plot_fig5(data, dataname):
    #ece, ens_acc, nll, nll_norm, totsubnet_acc, totsubnet_nll
    nll = data[:5,2]
    ensemble_accuracy = data[:5,1]
    totsubnet_acc = data[:5,4]
    totsubnet_nll = data[:5,5]
    x = range(1,6)
    
    plt.plot(x, ensemble_accuracy,marker='o',label = "Ensemble")
    plt.plot(x, totsubnet_acc, linestyle='--', marker='o', label = "Subnetworks")
    plt.xlabel('M')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy (%) on ' + dataname)
    plt.legend()
    plt.show()
    plt.savefig('figs/fig5_acc_' + dataname)

    plt.plot(x, -nll, marker='o' ,label = "Ensemble")
    plt.plot(x, -totsubnet_nll, linestyle='--', marker='o', label = "Subnetworks")
    plt.xlabel('M')
    plt.ylabel('Likelihood')
    plt.title('Test Log-likelihood on ' + dataname)
    plt.legend()
    plt.show()
    plt.savefig('figs/fig5_likelihood' + dataname)

    return None


if __name__ == '__main__':
    fig5_cifar100 = np.load('experiment_results/fig5_experiment_results_cifar100.npy')
    fig5_cifar10 = np.load('experiment_results/fig5_experiment_results_cifar10.npy')
    plot_fig5(fig5_cifar10, 'Cifar 10')
    plot_fig5(fig5_cifar100, 'Cifar 100')
