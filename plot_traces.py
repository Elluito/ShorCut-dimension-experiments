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


def plot_list(list_of_names=[], legend=[], title=""):
    plt.title(title, fontsize=20)
    color = cm.rainbow(np.linspace(0, 1, len(list_of_names)))

    for i, key in enumerate(list_of_names):
        c = color[i]
        plt.plot(running_mean(500, np.loadtxt(key)), c=c)
    plt.legend(legend, prop={"size": 20})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()


def plot_mean_and_CI(t, mean, lb, ub, color_mean=None, color_shading=None, label=""):
    # plot the shaded range of the confidence intervals
    plt.fill_between(t, ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(t, mean, c=color_mean, label=label)


def plot_list_with_x_axis(list_of_names=[], legend=[], x_axises=[], normalizer=1, x_axis_name="", title=""):
    plt.title(title, fontsize=20)
    color = cm.rainbow(np.linspace(0, 1, len(list_of_names)))

    for i, key in enumerate(list_of_names):
        c = color[i]
        x = running_mean(500, np.loadtxt(x_axises[i])) / normalizer
        plt.plot(np.cumsum(x), running_mean(500, np.loadtxt(key)), c=c)
    plt.legend(legend, prop={"size": 20})
    plt.xticks(fontsize=15)
    plt.xlabel(x_axis_name, fontsize=20)
    plt.yticks(fontsize=15)
    plt.show()


def plot_traces():
    loss_regularized = {-2: np.loadtxt("traces/loss_normal_training.txt")}
    loss_regularized.update({-1: np.loadtxt("traces/loss_pruned_training.txt")})
    for file in gb.glob("traces/loss_restricted_training*"):
        temp = np.loadtxt(file)

        distance = re.search("(?<=value\_)(.*?)(?=\.t)", file).group()
        loss_regularized.update({float(distance): temp})

    test_regularized = {-2: np.loadtxt("traces/test_normal_training.txt")}
    test_regularized.update({-1: np.loadtxt("traces/test_pruned_training.txt")})
    for file in gb.glob("traces/test_restricted_training*"):
        temp = np.loadtxt(file)
        distance = re.search("(?<=value\_)(.*?)(?=\.t)", file).group()
        test_regularized.update({float(distance): temp})

        print(file)
    # test_regularized.update({20000.0:np.loadtxt("traces/test_krone_training.txt")})
    # loss_regularized.update({20000.0:np.loadtxt("traces/loss_krone_training.txt")})
    legend = []
    # distances.sort()

    plt.title("Loss during training in CIFAR 10", fontsize=20)
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
    plt.plot(running_mean(500, np.loadtxt("traces/loss_krone_pruned_training.txt")), c="b")
    plt.legend(legend, prop={"size": 20})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.figure()

    legend = []
    plt.title("Accuracy in test set of CIFAR10", fontsize=20)

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
    plt.plot(running_mean(500, np.loadtxt("traces/test_krone_pruned_training.txt")), c="b")
    plt.legend(legend, prop={"size": 20})
    # plt.legend(legend)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()


def historgram_3d(dictionary, comparison="all"):
    # Fixing random state for reproducibility
    np.random.seed(19680801)
    cumulative = None
    comparison_by_layer = {}
    for layer_name, layer in list(dictionary.items()):
        cumulative_for_layer = []
        for epoch, iterations_dictionary in sorted(layer.items()):
            for iteration, weights in sorted(iterations_dictionary.items()):
                cumulative_for_layer.append(weights)
        comparison_by_layer[layer_name] = cumulative_for_layer
        if cumulative is None:
            cumulative = np.array(cumulative_for_layer)
        else:
            cumulative = np.hstack((cumulative, np.array(cumulative_for_layer)))
    if comparison == "all":
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x, y = np.random.rand(2, 100) * 4
        histograms = []
        bins = [0.1, 0.01, 0.001, 0.0001, 0.00001, -0.00001, -0.0001, -0.001, -0.01, -0.1]
        bins.reverse()
        for elem in cumulative:
            hist, _ = np.histogram(elem, bins)
            histograms.append(hist)

        histograms = np.array(histograms)
        x_edge = np.array(list(range(1, 11)))
        y_edge = np.array(bins)
        hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])

        # Construct arrays for the anchor positions of the 16 bars.
        xpos, ypos = np.meshgrid(x_edge, y_edge[:-1] + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        # Construct arrays with the dimensions for the 16 bars.
        dx = dy = 0.001 * np.ones_like(zpos)
        dz = histograms.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

        plt.show()
    if comparison == "layers":
        import tensorflow as tf
        path_colab = "/content/drive/MyDrive/Colab Notebooks/Extra-dimension-role/"
        w = tf.summary.create_file_writer(f'{path_colab}histograms/logs')
        for name, values in comparison_by_layer.items():
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # histograms = []
            # bins = [0.1, 0.01, 0.001, 0.0001, 0.00001, -0.00001, -0.0001, -0.001, -0.01, -0.1]
            # bins.reverse()

            # x_edge = np.array(list(range(1, 11)))
            # y_edge = np.array(bins)
            #
            # # Construct arrays for the anchor positions of the 16 bars.
            # xpos, ypos = np.meshgrid(x_edge, y_edge[:-1] , indexing="xy")
            # xpos = xpos.ravel()
            # ypos = ypos.ravel()
            # zpos = 0
            #
            # # Construct arrays with the dimensions for the 16 bars.
            # dx = 0.001 * np.ones_like(zpos)
            # dz = histograms.ravel()
            # dy = 0.0001 * np.ones_like(zpos)
            # ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
            # plt.title(name, fontsize=15)
            # plt.show()

            with w.as_default():
                for step, elem in enumerate(values):
                    # Generate fake "activations".
                    tf.summary.histogram(f"layer_{name}", elem, step=step)
                    # tf.summary.histogram("layer2/activate", activations[1], step=step)
                    # tf.summary.histogram("layer3/activate", activations[2], step=step)

    else:
        raise Exception(f"Mode {comparison} is not supported.")


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
    ax1.hist(m_weigths, 1000)
    ax2.set_title("Weigths that don't belong to the mask")
    ax2.hist(no_m_weigths, 1000)
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
    stast_, p_value = stats.ttest_ind(m_weigths, no_m_weigths)
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
    # names = ["traces/loss_training_KFAC_small.txt","traces/loss_training_SGD_big.txt",
    #          "traces/loss_training_SGD_small.txt",
    #          "traces/loss_training_KFAC_big.txt"]
    # legend = ["KFAC small network","SGD large network","SGD large network","KFAC large network"]
    # plot_list(names,legend,"Loss for CIFAR10")
    #
    #
    # names = ["pure_experiments/loss_training_FC_small_KFAC.txt","pure_experiments/loss_training_FC_small_SAM.txt", "pure_experiments/loss_training_FC_small_SGD.txt","pure_experiments/loss_training_FC_big_SGD.txt"]; x_axis = ["pure_experiments/function_call_FC_small_KFAC.txt","pure_experiments/function_call_FC_small_SAM.txt", "pure_experiments/function_call_FC_small_SGD.txt","pure_experiments/function_call_FC_big_SGD.txt"]
    # norm = 1
    # if "time" in x_axis[0]:
    #     norm = 1e9
    # legend = ["KFAC small network","SAM small network","SGD small network","SGD large network"]
    # plot_list_with_x_axis(names,legend,x_axis,norm,"$f_n$","Loss function for CIFAR10 for FC Network")
    #
    #
    #
    # names = ["pure_experiments/test_training_FC_small_KFAC.txt","pure_experiments/test_training_FC_small_SAM.txt", "pure_experiments/test_training_FC_small_SGD.txt","pure_experiments/test_training_FC_big_SGD.txt"]; x_axis = ["pure_experiments/function_call_FC_small_KFAC.txt","pure_experiments/function_call_FC_small_SAM.txt", "pure_experiments/function_call_FC_small_SGD.txt","pure_experiments/function_call_FC_big_SGD.txt"]
    # # legend = ["SGD with large ConvNet","KFAC with large ConvNet","KFAC with small ConvNet","SAM with small ConvNet"]
    #
    # plot_list_with_x_axis(names,legend,x_axis,norm,"$f_n$","Test accuracy for CIFAR10 for FC Network")

    # names = ["traces/test_training_SGD_conv_big.txt","traces/test_training_KFAC_conv_big.txt",
    #          "traces/test_training_KFAC_conv_small.txt","traces/test_training_SAM_conv_small.txt"]

    ############################################################################################
    # FAIR COMPARISONS
    # names = ["traces2/loss_training_KFAC_small.txt", "traces2/loss_training_SGD_big.txt",
    #          "traces2/loss_training_SGD_small.txt",
    #          "traces2/loss_training_KFAC_big.txt"]
    # legend = ["KFAC small network", "SGD large network", "SGD large network", "KFAC large network"]
    # plot_list(names, legend, "Loss for CIFAR10")
    #
    # names = ["traces2/test_training_KFAC_conv_small.txt", "traces2/test_training_SGD_conv_big.txt",
    #          "traces2/test_training_SGD_conv_small.txt"]
    # # ,"traces2/test_training_KFAC_conv_big.txt"]
    # function_cals = ["traces2/function_call_KFAC_conv_small.txt", "traces2/function_calls_SGD_big.txt",
    #                  "traces2/function_call_SGD_conv_small.txt", "trances2/function_call_SAM_conv_small.txt"]
    # # ,"traces2/test_training_KFAC_big.txt"]
    # time = ["traces2/time_KFAC_conv_small.txt", "traces2/time_SGD_conv_big.txt",
    #         "traces2/time_SGD_conv_small.txt", "traces2/time_SAM_conv_small.txt"]
    # legend = ["KFAC small network", "SGD large network", "SGD small network", "SAM small network"]
    # l = np.loadtxt("traces2/time_KFAC_conv_small.txt")
    #
    # plot_list_with_x_axis(names, legend, time, 1e9, "Computation time", "Test accuracy for CIFAR10")

    ###### 3D histograms ########
    from main import read_regitered_weigths

    X = read_regitered_weigths("net_model/SGD_pruned_test/", type="weigth")
    G = read_regitered_weigths("net_model/SGD_pruned_test/", type="gradient")
    historgram_3d(X, "layers")
