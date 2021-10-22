import matplotlib.pyplot as plt
from torch.nn.utils import prune
import pandas as pd
import numpy as np
import glob as gb
from matplotlib.pyplot import cm
import re
import torch
from main import Net, get_inverted_mask, count_parameters

loss_regularized = {-1: np.loadtxt("traces/loss_normal_training.txt")}
distances = []


def running_mean(N, x):
    return np.convolve(x, np.ones((N,)) / N, mode='valid')

def plot_list(list_of_names=[],legend=[],title=""):

    plt.title(title, fontsize=20)
    color = cm.rainbow(np.linspace(0, 1, len(list_of_names)))

    for i, key in enumerate(list_of_names):
        c = color[i]
        plt.plot(running_mean(500, np.loadtxt(key)), c=c)
    plt.legend(legend, prop={"size": 20})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

def plot_list_with_x_axis(list_of_names=[],legend=[],x_axises =[],x_axis_name="",title=""):
    plt.title(title, fontsize=20)
    color = cm.rainbow(np.linspace(0, 1, len(list_of_names)))

    for i, key in enumerate(list_of_names):
        c = color[i]
        x = running_mean(500,np.loadtxt(x_axises))
        plt.plot(np.cumsum(x),running_mean(500, np.loadtxt(key)), c=c)
    plt.legend(legend, prop={"size": 20})
    plt.xticks(fontsize=15)
    plt.xlabel(x_axis_name,fontsize=20)
    plt.yticks(fontsize=15)
    plt.show()


def plot_traces():

    loss_regularized = {-2: np.loadtxt("traces/loss_normal_training.txt")}
    loss_regularized.update({-1: np.loadtxt("traces/loss_pruned_training.txt")})
    for file in gb.glob("traces/loss_restricted_training*"):
        temp = np.loadtxt(file)

        distance = re.search("(?<=value\_)(.*?)(?=\.t)",file).group()
        loss_regularized.update({float(distance): temp})

    test_regularized = {-2: np.loadtxt("traces/test_normal_training.txt")}
    test_regularized.update({-1: np.loadtxt("traces/test_pruned_training.txt")})
    for file in gb.glob("traces/test_restricted_training*"):
        temp = np.loadtxt(file)
        distance = re.search("(?<=value\_)(.*?)(?=\.t)",file).group()
        test_regularized.update({float(distance): temp})

        print(file)
    # test_regularized.update({20000.0:np.loadtxt("traces/test_krone_training.txt")})
    # loss_regularized.update({20000.0:np.loadtxt("traces/loss_krone_training.txt")})
    legend = []
    # distances.sort()

    plt.title("Loss during training in CIFAR 10",fontsize=20)
    color = cm.rainbow(np.linspace(0, 1, len(test_regularized.values())))

    for i, key in enumerate(sorted(loss_regularized)):
        c = color[i]
        plt.plot(running_mean(500, loss_regularized[key]), c=c)
        if key == -2:
            legend.append("$L=\infty$")
        if key == -1:
            legend.append("Pruned from scratch")
        elif key != -1 and key != -2:
            legend.append(f"$L=${key}")
    legend.append("Kronecker-pruned")
    plt.plot(running_mean(500,np.loadtxt("traces/loss_krone_pruned_training.txt")),c="b")
    plt.legend(legend,prop={"size":20})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.figure()

    legend = []
    plt.title("Accuracy in test set of CIFAR10",fontsize=20)

    for i, key in enumerate(sorted(test_regularized)):
        c = color[i]
        plt.plot(running_mean(500, test_regularized[key]), c=c)

        if key == -2:
            legend.append("$L=\infty$")
        if key == -1:
            legend.append("Pruned from scratch")
        elif key != -1 and key != -2:
            legend.append(f"$L=${key}")
    legend.append("Kronecker-pruned")
    plt.plot(running_mean(500,np.loadtxt("traces/test_krone_pruned_training.txt")),c="b")
    plt.legend(legend,prop={"size":20})
    # plt.legend(legend)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()


def histogram_of_model(number):
    model = Net()
    prune.global_unstructured(
        model.parameters_to_prune(),
        pruning_method=prune.L1Unstructured,
        amount=count_parameters(model) // 2,
    )

    model.load_state_dict(torch.load("model_trained_vanila_pruned"))
    inv_mask = get_inverted_mask(model)
    mask = list(model.buffers())
    real_model = Net()
    real_model.load_state_dict(torch.load(f"model_restricted_trained_{number}"))
    j = 0
    m_weigths = []
    no_m_weigths = []
    for (name, param) in real_model.named_parameters():
        if 'bias' not in name:
            thing = param[mask[j].type(torch.BoolTensor)]
            anti_thing = param[inv_mask[j].type(torch.BoolTensor)]
            m_weigths.extend(thing.detach().numpy().flatten())
            if number <= 0:
                print(param[inv_mask[j].type(torch.BoolTensor)])
            no_m_weigths.extend(anti_thing.detach().numpy().flatten())
            j += 1
    fig, [ax1, ax2] = plt.subplots(1, 2, sharex=True)
    ax1.set_title("Weigths that belong to the mask")
    ax1.hist(m_weigths,1000)
    ax2.set_title("Weigths that don't belong to the mask")
    ax2.hist(no_m_weigths,1000)
    plt.savefig(f"Images/hitogram_model_l_{number}.jpg")

def t_student_test(number):
    model = Net()
    prune.global_unstructured(
        model.parameters_to_prune(),
        pruning_method=prune.L1Unstructured,
        amount=count_parameters(model) // 2,
    )

    model.load_state_dict(torch.load("model_trained_vanila_pruned"))
    inv_mask = get_inverted_mask(model)
    mask = list(model.buffers())
    real_model = Net()
    real_model.load_state_dict(torch.load(f"model_trained_lambda_{number}"))

    j = 0
    m_weigths = []
    no_m_weigths = []
    for (name, param) in real_model.named_parameters():
        if 'bias' not in name:
            thing = param[mask[j].type(torch.BoolTensor)]
            anti_thing = param[inv_mask[j].type(torch.BoolTensor)]
            m_weigths.extend(thing.detach().numpy().flatten())
            no_m_weigths.extend(anti_thing.detach().numpy().flatten())
            j += 1
    from scipy import stats
    stast_,p_value = stats.ttest_ind(m_weigths,no_m_weigths)
    print(f"Model number {number}")
    print(f"T-Statistic: {stast_}")
    print(f"P-value: {p_value}")


if __name__ == '__main__':
    # n = [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # for i in n:
    #     t_student_test(i)
    # model = Net()
    # prune.global_unstructured(
    #     model.parameters_to_prune(),
    #     pruning_method=prune.L1Unstructured,
    #     amount=count_parameters(model) // 2,
    # )
    # print("Original model")
    # model.load_state_dict(torch.load("model_trained_vanila_pruned"))
    # inv_mask = get_inverted_mask(model)
    # mask = list(model.buffers())
    # j = 0
    # m_weigths = []
    # no_m_weigths = []
    # for (name, param) in model.named_parameters():
    #     if 'bias' not in name:
    #         thing = param[mask[j].type(torch.BoolTensor)]
    #         anti_thing = param[inv_mask[j].type(torch.BoolTensor)]
    #         m_weigths.extend(thing.detach().numpy().flatten())
    #         no_m_weigths.extend(anti_thing.detach().numpy().flatten())
    #         j += 1
    # from scipy import stats
    # stast_,p_value = stats.ttest_ind(m_weigths,no_m_weigths)
    # print(f"T-Statistic: {stast_}")
    # print(f"P-value: {p_value}")
    # plot_traces()
    # distances = [80, 10, 5, 2, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0]
    # distances = [0]
    # for i in distances:
    #     histogram_of_model(i)
    names = ["traces/loss_training_KFAC_small.txt","traces/loss_training_SGD_big.txt",
             "traces/loss_training_SGD_small.txt",
             "traces/loss_training_KFAC_big.txt"]
    legend = ["KFAC small network","SGD large network","SGD large network","KFAC large network"]
    plot_list(names,legend,"Loss for CIFAR10")


    names = ["traces/test_training_KFAC_small.txt","traces/test_training_SGD_big.txt",
             "traces/test_training_SGD_small.txt",
            "traces/test_training_KFAC_big.txt"]
    legend = ["KFAC small network","SGD large network","SGD small network","KFAC large network"]
    plot_list(names,legend,"Test accuracy for CIFAR10")
    # names = ["traces/loss_training_SGD_conv_big.txt","traces/loss_training_KFAC_conv_big.txt",
    #          "traces/loss_training_KFAC_conv_small.txt","traces/loss_training_SAM_conv_small.txt"]
    # legend = ["SGD with large ConvNet","KFAC with large ConvNet","KFAC with small ConvNet","SAM with small ConvNet"]
    #
    # plot_list(names,legend,"Loss function for CIFAR10")
    #
    # names = ["traces/test_training_SGD_conv_big.txt","traces/test_training_KFAC_conv_big.txt",
    #          "traces/test_training_KFAC_conv_small.txt","traces/test_training_SAM_conv_small.txt"]
