import numpy as np
import pandas as pd
import json
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

def plot_fig6(data):
    df = pd.DataFrame.from_dict(data)

    print(df)

    reps_1 = df.loc[0:2,'results']
    reps_2 = df.loc[3:5,'results']
    reps_3 = df.loc[6:8,'results']
    
    # Accuracy
    data1 = [reps_1.iloc[0][1], reps_1.iloc[1][1], reps_1.iloc[2][1]]
    data2 = [reps_2.iloc[0][1], reps_2.iloc[1][1], reps_2.iloc[2][1]]
    data3 = [reps_3.iloc[0][1], reps_3.iloc[1][1], reps_3.iloc[2][1]]

    data = [data1, data2, data3]

    boxplot = plt.boxplot(data, patch_artist=True, vert=True)
    plt.title('Test Accuarcy (%)')
    plt.xlabel('# batch repetitions')
    plt.ylabel('Accuracy')
    plt.grid(True)

    colors = ['blue', 'orange', 'green']

    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.show()
    plt.savefig('figs/fig6_acc_Cifar10')

    # Log lik
    data1 = [-reps_1.iloc[0][2], -reps_1.iloc[1][2], -reps_1.iloc[2][2]]
    data2 = [-reps_2.iloc[0][2], -reps_2.iloc[1][2], -reps_2.iloc[2][2]]
    data3 = [-reps_3.iloc[0][2], -reps_3.iloc[1][2], -reps_3.iloc[2][2]]

    data = [data1, data2, data3]

    boxplot = plt.boxplot(data, patch_artist=True, vert=True)
    plt.title('Test Log-likelihood')
    plt.xlabel('# batch repetitions')
    plt.ylabel('Log-likelihood')
    plt.grid(True)

    colors = ['blue', 'orange', 'green']

    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.show()
    plt.savefig('figs/fig6_loglikelihood_Cifar10')

    return None

if __name__ == '__main__':
    
    fig5_cifar100 = np.load('experiment_results/fig5_experiment_results_cifar100.npy')
    fig5_cifar10 = np.load('experiment_results/fig5_experiment_results_cifar10.npy')
    f = open('experiment_results/fig6_dict.json',)
    fig6_cifar10 = json.load(f)

    plot_fig5(fig5_cifar10, 'Cifar 10')
    plot_fig5(fig5_cifar100, 'Cifar 100')
    plot_fig6(fig6_cifar10)