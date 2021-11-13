import glob
import re
import matplotlib.pyplot as plt
import time
import json
import torch.nn as nn
from torch.nn import Flatten
import os
from itertools import cycle
import torchvision
import torchvision.transforms as transforms
from ignite.metrics import Accuracy
import torch
from hessian_eigenthings import compute_hessian_eigenthings
from torch.nn import Parameter

# from KFAC_Pytorch.optimizers.kfac import KFACOptimizer

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
accuracy = Accuracy()
from definitions import LinearMasked, Conv2dMasked
from definitions import MaskedModule
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from KFAC_Pytorch.optimizers import KFACOptimizer
from sam import SAM
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


class ModelPytorch(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ModelPytorch, self).__init__()
        temp_conv1 = nn.Conv2d(3, 64, 3)

        weigth_mask = torch.ones_like(temp_conv1.weight)
        temp_conv2 = nn.Conv2d(64, 128, 3)
        weigth_mask2 = torch.ones_like(temp_conv2.weight)
        self.Conv1 = Conv2dMasked(temp_conv1, weigth_mask)
        self.Conv2 = Conv2dMasked(temp_conv2, weigth_mask2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        temp_linear1 = nn.Linear(4608, 516)
        temp_linear2 = nn.Linear(516, 10)
        weigth_mask_linear = torch.ones_like(temp_linear1.weight)
        weigth_mask2_linear = torch.ones_like(temp_linear2.weight)
        self.linear1 = LinearMasked(temp_linear1, weigth_mask_linear)
        self.linear2 = LinearMasked(temp_linear2, weigth_mask2_linear)
        # self.add_module("conv1",self.Conv1)
        # self.add_module("conv2",self.Conv2)
        # self.add_module("linear1",self.linear1)
        # self.add_module("linear2",self.linear2)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        rep = nn.functional.relu(self.Conv1(x))
        rep = self.maxpool1(rep)
        rep = nn.functional.relu(self.Conv2(rep))
        rep = self.maxpool2(rep)
        rep = nn.Flatten()(rep)

        h_relu = self.linear1(rep).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

    def parameters_to_prune(self):
        weights = ["weight"] * 4
        layers = [self.Conv1, self.Conv2, self.linear1, self.linear2]
        return list(zip(layers, weights))

    def ensure_device(self, device):
        self.Conv1.to(device)
        self.Conv2.to(device)
        self.linear1.to(device)
        self.linear2.to(device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NewSmallNet(nn.Module):
    def __init__(self):
        super(NewSmallNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(4608, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = self.fc1(x)
        return x

    def parameters_to_prune(self):
        layers = []
        for module in self.modules():
            if not isinstance(module, NewSmallNet):
                layers.append(module)
        weights = ["weights"] * len(layers)
        return list(zip(layers, weights))

    def ensure_device(self, device):
        for module in self.modules():
            if not isinstance(module, NewSmallNet):
                module.to(device)

    def apply_buffers(self):
        masks = list(self.buffers())
        for (name, param) in self.named_parameters():
            if 'bias' not in name:
                param = param * masks[i].cuda()
                i += 1
        return True

    def partial_grad(self, data, target, loss_function):
        """
        Function to compute the grad
        args : data, target, loss_function
        return loss
        """
        outputs = self.forward(data)
        loss = loss_function(outputs, target)
        loss.backward()  # compute grad
        return loss

    def calculate_loss_grad(self, dataset, loss_function, n_samples):
        """
        Function to compute the full loss and the full gradient
        args : dataset, loss function and number of samples
        return : total loss and full grad norm
        """
        total_loss = 0.0

        for i_grad, data_grad in enumerate(dataset):
            inputs, labels = data_grad
            inputs, labels = Variable(inputs), Variable(labels)  # wrap data and target into variable
            total_loss += (1. / n_samples) * self.partial_grad(inputs, labels, loss_function).data[0]

        return total_loss

    def constricted_train(self, dataset, loss_function, n_epoch, learning_rate, momentum, iter_testloader,
                          file_name_sufix, inverse_mask,
                          restriction, optimizer="SGD"):
        """
        Function to updated weights with a SVRG backpropagation
        args : dataset, loss function, number of epochs, learning rate
        return : total_loss_epoch, grad_norm_epoch
        """
        if optimizer == "SGD":
            open(file_name_sufix + f"/test_restricted_training_value_{restriction}.txt", "w").close()
            open(file_name_sufix + f"/loss_restricted_training_value_{restriction}.txt", "w").close()
            velocity_buffer = {}
            dampening = 0
            nesterov = 0
            for epoch in range(n_epoch):
                running_loss = 0.0
                # previous_net_sgd = copy.deepcopy(self)  # update previous_net_sgd
                # previous_net_grad = copy.deepcopy(self)  # update previous_net_grad
                #
                # # Compute full grad
                # previous_net_grad.zero_grad()  # grad = 0
                #
                # Run over the dataset
                for i_data, data in enumerate(dataset):
                    inputs, labels = data
                    inputs, labels = inputs.cuda(), labels.cuda()  # wrap data and target into variable
                    # inputs.cuda()
                    # labels.cuda()
                    # Compute cur stoc grad
                    self.zero_grad()  # grad = 0
                    cur_loss = self.partial_grad(inputs, labels, loss_function)

                    # Backward
                    j = 0
                    for name, param in self.named_parameters():
                        d_p = param.grad.data
                        update_value = 0
                        if name not in velocity_buffer.keys():
                            buf = velocity_buffer[name] = torch.zeros_like(d_p)
                            buf.mul_(momentum).add_(d_p)

                        else:
                            buf = velocity_buffer[name]
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                        if "bias" not in name:
                            param.data.add_(-learning_rate, d_p)
                            with torch.no_grad():
                                param.data[inverse_mask[j].type(
                                    torch.BoolTensor)].clamp_(min=-restriction, max=restriction)
                            j += 1
                        else:
                            param.data.add_(-learning_rate, d_p)
                    # print statistics
                    running_loss += cur_loss.item()
                    with open(file_name_sufix + f"/loss_restricted_training_value_{restriction}.txt", "a") as f:
                        f.write(f"{cur_loss.item()}\n")
                    eval = evaluate_model(self, None, iter_testloader, partial=True)
                    with open(file_name_sufix + f"/test_restricted_training_value_{restriction}.txt", "a") as f:
                        f.write(f"{eval}\n")
                    if i_data % 200 == 199:  # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i_data + 1, running_loss / 2000))
                        running_loss = 0.0
        if optimizer == "kronecker":

            open(file_name_sufix + f"/test_krone_restricted_training_value_{restriction}.txt", "w").close()
            open(file_name_sufix + f"/loss_krone_restricted_training_value_{restriction}.txt", "w").close()
            optimizer = KFACOptimizer(self)
            criterion = loss_function
            for epoch in range(n_epoch):
                running_loss = 0.0
                # Run over the dataset
                for i_data, data in enumerate(dataset):
                    inputs, labels = data
                    inputs, labels = inputs.cuda(), labels.cuda()  # wrap data and target into variable
                    # inputs.cuda()
                    # labels.cuda()
                    # Compute cur stoc grad
                    # self.zero_grad()  # grad = 0
                    # cur_loss = self.partial_grad(inputs, labels, loss_function)
                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    if optimizer.steps % optimizer.TCov == 0:
                        # compute true fisher
                        optimizer.acc_stats = True
                        with torch.no_grad():
                            sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1),
                                                          1).squeeze().cuda()
                        loss_sample = criterion(outputs, sampled_y)
                        loss_sample.backward(retain_graph=True)
                        optimizer.acc_stats = False
                        optimizer.zero_grad()  # clear the gradient for computing true-fisher.
                    loss.backward()
                    optimizer.step()
                    item = loss.item()
                    running_loss += item
                    with open(file_name_sufix + f"/loss_krone_restricted_training_value_{restriction}.txt", "a") as f:
                        f.write(f"{item}\n")
                    eval = evaluate_model(self, None, iter_testloader, partial=True)
                    with open(file_name_sufix + f"/test_krone_restricted_training_value_{restriction}.txt",
                              "a") as f:
                        f.write(f"{eval}\n")
                    if i_data % 200 == 199:  # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i_data + 1, running_loss / 200))
                        running_loss = 0.0


class NewNet(nn.Module):
    def __init__(self):
        super(NewNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(3, 64, 3, padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 128 * 2, 3, padding=(1, 1))
        self.conv4 = nn.Conv2d(128 * 2, 128 * 4, 3)
        self.fc1 = nn.Linear(18432, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = self.fc1(x)
        return x

    def parameters_to_prune(self):
        layers = []
        for module in self.modules():
            if not isinstance(module, NewSmallNet):
                layers.append(module)
        weights = ["weight"] * len(layers)
        return list(zip(layers, weights))

    def ensure_device(self, device):
        for module in self.modules():
            if not isinstance(module, NewNet):
                module.to(device)

    def apply_buffers(self):
        masks = list(self.buffers())
        for (name, param) in self.named_parameters():
            if 'bias' not in name:
                param = param * masks[i].cuda()
                i += 1
        return True

    def partial_grad(self, data, target, loss_function):
        """
        Function to compute the grad
        args : data, target, loss_function
        return loss
        """
        outputs = self.forward(data)
        loss = loss_function(outputs, target)
        loss.backward()  # compute grad
        return loss

    def calculate_loss_grad(self, dataset, loss_function, n_samples):
        """
        Function to compute the full loss and the full gradient
        args : dataset, loss function and number of samples
        return : total loss and full grad norm
        """
        total_loss = 0.0

        for i_grad, data_grad in enumerate(dataset):
            inputs, labels = data_grad
            inputs, labels = Variable(inputs), Variable(labels)  # wrap data and target into variable
            total_loss += (1. / n_samples) * self.partial_grad(inputs, labels, loss_function).data[0]

        return total_loss

    def constricted_train(self, dataset, loss_function, n_epoch, learning_rate, momentum, iter_testloader,
                          file_name_sufix, inverse_mask,
                          restriction, optimizer="SGD"):
        """
        Function to updated weights with a SVRG backpropagation
        args : dataset, loss function, number of epochs, learning rate
        return : total_loss_epoch, grad_norm_epoch
        """
        if optimizer == "SGD":
            open(file_name_sufix + f"/test_restricted_training_value_{restriction}.txt", "w").close()
            open(file_name_sufix + f"/loss_restricted_training_value_{restriction}.txt", "w").close()
            velocity_buffer = {}
            dampening = 0
            nesterov = 0
            for epoch in range(n_epoch):
                running_loss = 0.0
                # previous_net_sgd = copy.deepcopy(self)  # update previous_net_sgd
                # previous_net_grad = copy.deepcopy(self)  # update previous_net_grad
                #
                # # Compute full grad
                # previous_net_grad.zero_grad()  # grad = 0
                #
                # Run over the dataset
                for i_data, data in enumerate(dataset):
                    inputs, labels = data
                    inputs, labels = inputs.cuda(), labels.cuda()  # wrap data and target into variable
                    # inputs.cuda()
                    # labels.cuda()
                    # Compute cur stoc grad
                    self.zero_grad()  # grad = 0
                    cur_loss = self.partial_grad(inputs, labels, loss_function)

                    # Backward
                    j = 0
                    for name, param in self.named_parameters():
                        d_p = param.grad.data
                        update_value = 0
                        if name not in velocity_buffer.keys():
                            buf = velocity_buffer[name] = torch.zeros_like(d_p)
                            buf.mul_(momentum).add_(d_p)

                        else:
                            buf = velocity_buffer[name]
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                        if "bias" not in name:
                            param.data.add_(-learning_rate, d_p)
                            with torch.no_grad():
                                param.data[inverse_mask[j].type(
                                    torch.BoolTensor)].clamp_(min=-restriction, max=restriction)
                            j += 1
                        else:
                            param.data.add_(-learning_rate, d_p)
                    # print statistics
                    running_loss += cur_loss.item()
                    with open(file_name_sufix + f"/loss_restricted_training_value_{restriction}.txt", "a") as f:
                        f.write(f"{cur_loss.item()}\n")
                    eval = evaluate_model(self, None, iter_testloader, partial=True)
                    with open(file_name_sufix + f"/test_restricted_training_value_{restriction}.txt", "a") as f:
                        f.write(f"{eval}\n")
                    if i_data % 200 == 199:  # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i_data + 1, running_loss / 2000))
                        running_loss = 0.0
        if optimizer == "kronecker":

            open(file_name_sufix + f"/test_krone_restricted_training_value_{restriction}.txt", "w").close()
            open(file_name_sufix + f"/loss_krone_restricted_training_value_{restriction}.txt", "w").close()
            optimizer = KFACOptimizer(self)
            criterion = loss_function
            for epoch in range(n_epoch):
                running_loss = 0.0
                # Run over the dataset
                for i_data, data in enumerate(dataset):
                    inputs, labels = data
                    inputs, labels = inputs.cuda(), labels.cuda()  # wrap data and target into variable
                    # inputs.cuda()
                    # labels.cuda()
                    # Compute cur stoc grad
                    # self.zero_grad()  # grad = 0
                    # cur_loss = self.partial_grad(inputs, labels, loss_function)
                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    if optimizer.steps % optimizer.TCov == 0:
                        # compute true fisher
                        optimizer.acc_stats = True
                        with torch.no_grad():
                            sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1),
                                                          1).squeeze().cuda()
                        loss_sample = criterion(outputs, sampled_y)
                        loss_sample.backward(retain_graph=True)
                        optimizer.acc_stats = False
                        optimizer.zero_grad()  # clear the gradient for computing true-fisher.
                    loss.backward()
                    optimizer.step()
                    item = loss.item()
                    running_loss += item
                    with open(file_name_sufix + f"/loss_krone_restricted_training_value_{restriction}.txt", "a") as f:
                        f.write(f"{item}\n")
                    eval = evaluate_model(self, None, iter_testloader, partial=True)
                    with open(file_name_sufix + f"/test_krone_restricted_training_value_{restriction}.txt",
                              "a") as f:
                        f.write(f"{eval}\n")
                    if i_data % 200 == 199:  # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i_data + 1, running_loss / 200))
                        running_loss = 0.0


class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(4608, 258)
        self.fc2 = nn.Linear(258, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def parameters_to_prune(self):
        weights = ["weight"] * 4
        layers = [self.conv1, self.conv2, self.fc1, self.fc2]
        return list(zip(layers, weights))

    def ensure_device(self, device):
        self.conv1.to(device)
        self.conv2.to(device)
        self.fc1.to(device)
        self.fc2.to(device)

    def apply_buffers(self):
        masks = list(self.buffers())
        for (name, param) in self.named_parameters():
            if 'bias' not in name:
                param = param * masks[i].cuda()
                i += 1
        return True

    def partial_grad(self, data, target, loss_function):
        """
        Function to compute the grad
        args : data, target, loss_function
        return loss
        """
        outputs = self.forward(data)
        loss = loss_function(outputs, target)
        loss.backward()  # compute grad
        return loss

    def calculate_loss_grad(self, dataset, loss_function, n_samples):
        """
        Function to compute the full loss and the full gradient
        args : dataset, loss function and number of samples
        return : total loss and full grad norm
        """
        total_loss = 0.0

        for i_grad, data_grad in enumerate(dataset):
            inputs, labels = data_grad
            inputs, labels = Variable(inputs), Variable(labels)  # wrap data and target into variable
            total_loss += (1. / n_samples) * self.partial_grad(inputs, labels, loss_function).data[0]

        return total_loss

    def constricted_train(self, dataset, loss_function, n_epoch, learning_rate, momentum, iter_testloader,
                          file_name_sufix, inverse_mask,
                          restriction, optimizer="SGD"):
        """
        Function to updated weights with a SVRG backpropagation
        args : dataset, loss function, number of epochs, learning rate
        return : total_loss_epoch, grad_norm_epoch
        """
        if optimizer == "SGD":
            open(file_name_sufix + f"/test_restricted_training_value_{restriction}.txt", "w").close()
            open(file_name_sufix + f"/loss_restricted_training_value_{restriction}.txt", "w").close()
            velocity_buffer = {}
            dampening = 0
            nesterov = 0
            for epoch in range(n_epoch):
                running_loss = 0.0
                # previous_net_sgd = copy.deepcopy(self)  # update previous_net_sgd
                # previous_net_grad = copy.deepcopy(self)  # update previous_net_grad
                #
                # # Compute full grad
                # previous_net_grad.zero_grad()  # grad = 0
                #
                # Run over the dataset
                for i_data, data in enumerate(dataset):
                    inputs, labels = data
                    inputs, labels = inputs.cuda(), labels.cuda()  # wrap data and target into variable
                    # inputs.cuda()
                    # labels.cuda()
                    # Compute cur stoc grad
                    self.zero_grad()  # grad = 0
                    cur_loss = self.partial_grad(inputs, labels, loss_function)

                    # Backward
                    j = 0
                    for name, param in self.named_parameters():
                        d_p = param.grad.data
                        update_value = 0
                        if name not in velocity_buffer.keys():
                            buf = velocity_buffer[name] = torch.zeros_like(d_p)
                            buf.mul_(momentum).add_(d_p)

                        else:
                            buf = velocity_buffer[name]
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                        if "bias" not in name:
                            param.data.add_(-learning_rate, d_p)
                            with torch.no_grad():
                                param.data[inverse_mask[j].type(
                                    torch.BoolTensor)].clamp_(min=-restriction, max=restriction)
                            j += 1
                        else:
                            param.data.add_(-learning_rate, d_p)
                    # print statistics
                    running_loss += cur_loss.item()
                    with open(file_name_sufix + f"/loss_restricted_training_value_{restriction}.txt", "a") as f:
                        f.write(f"{cur_loss.item()}\n")
                    eval = evaluate_model(self, None, iter_testloader, partial=True)
                    with open(file_name_sufix + f"/test_restricted_training_value_{restriction}.txt", "a") as f:
                        f.write(f"{eval}\n")
                    if i_data % 200 == 199:  # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i_data + 1, running_loss / 2000))
                        running_loss = 0.0
        if optimizer == "kronecker":

            open(file_name_sufix + f"/test_krone_restricted_training_value_{restriction}.txt", "w").close()
            open(file_name_sufix + f"/loss_krone_restricted_training_value_{restriction}.txt", "w").close()
            optimizer = KFACOptimizer(self)
            criterion = loss_function
            for epoch in range(n_epoch):
                running_loss = 0.0
                # Run over the dataset
                for i_data, data in enumerate(dataset):
                    inputs, labels = data
                    inputs, labels = inputs.cuda(), labels.cuda()  # wrap data and target into variable
                    # inputs.cuda()
                    # labels.cuda()
                    # Compute cur stoc grad
                    # self.zero_grad()  # grad = 0
                    # cur_loss = self.partial_grad(inputs, labels, loss_function)
                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    if optimizer.steps % optimizer.TCov == 0:
                        # compute true fisher
                        optimizer.acc_stats = True
                        with torch.no_grad():
                            sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1),
                                                          1).squeeze().cuda()
                        loss_sample = criterion(outputs, sampled_y)
                        loss_sample.backward(retain_graph=True)
                        optimizer.acc_stats = False
                        optimizer.zero_grad()  # clear the gradient for computing true-fisher.
                    loss.backward()
                    optimizer.step()
                    item = loss.item()
                    running_loss += item
                    with open(file_name_sufix + f"/loss_krone_restricted_training_value_{restriction}.txt", "a") as f:
                        f.write(f"{item}\n")
                    eval = evaluate_model(self, None, iter_testloader, partial=True)
                    with open(file_name_sufix + f"/test_krone_restricted_training_value_{restriction}.txt",
                              "a") as f:
                        f.write(f"{eval}\n")
                    if i_data % 200 == 199:  # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i_data + 1, running_loss / 200))
                        running_loss = 0.0
def rename_attribute(object_, old_attribute_name, new_attribute_name):
    setattr(object_, new_attribute_name, getattr(object_, old_attribute_name))
    delattr(object_, old_attribute_name)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(3, 64, 3)
        rename_attribute(self.conv1,"weight","weight_orig")
        self.conv1.register_buffer("weight_mask",torch.ones_like(self.conv1.weight_orig.clone()))
        # self.conv1.register_buffer("conv1_mask",torch.ones_like(self.conv1.weight.clone()))
        self.conv2 = nn.Conv2d(64, 128, 3)
        rename_attribute(self.conv2,"weight","weight_orig")
        self.conv2.register_buffer("weight_mask",torch.ones_like(self.conv2.weight_orig.clone()))
        # self.conv2.register_buffer("conv2_mask",torch.ones_like(self.conv2.weight.clone()))
        self.fc1 = nn.Linear(4608, 516)
        rename_attribute(self.fc1,"weight","weight_orig")
        self.fc1.register_buffer("weight_mask",torch.ones_like(self.fc1.weight_orig.clone()))
        # self.fc1.register_buffer("w_mask",torch.ones_like(self.fc1.weight.clone()))
        self.fc2 = nn.Linear(516, 10)
        rename_attribute(self.fc2,"weight","weight_orig")
        self.fc2.register_buffer("weight_mask",torch.ones_like(self.fc2.weight_orig.clone()))
        # self.fc2.register_parameter("weigth_mask",nn.Parameter(torch.ones_like(self.fc2.weight.clone())))

        # model_dict = model.state_dict()
        #
        # # 1. filter out unnecessary keys
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # # 2. overwrite entries in the existing state dict
        # model_dict.update(pretrained_dict)
        # # 3. load the new state dict
        # model.load_state_dict(pretrained_dict)

    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            print(f"name {name}")
            own_state[name].copy_(param)
    def forward(self, x):
        x = F.max_pool2d(F.relu(F.conv2d(x,self.conv1.weight_orig,self.conv1.bias,self.conv1.stride,
                                         self.conv1.padding,self.conv1.dilation,self.conv1.groups)),(2, 2))
        x = F.max_pool2d(F.relu(F.conv2d(x,self.conv2.weight_orig,self.conv2.bias,self.conv2.stride, self.conv2.padding,self.conv2.dilation,self.conv2.groups)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(F.linear(x,self.fc1.weight_orig,self.fc1.bias))
        x = F.linear(x,self.fc2.weight_orig,self.fc2.bias)
        return x

    def parameters_to_prune(self):
        weights = ["weight"] * 4
        layers = [self.conv1, self.conv2, self.fc1, self.fc2]
        return list(zip(layers, weights))

    def ensure_device(self, device):
        self.conv1.to(device)
        self.conv2.to(device)
        self.fc1.to(device)
        self.fc2.to(device)

    def apply_mask(self, masks):
        # masks = list(self.buffers())
        i = 0
        for (name, param) in self.named_parameters():
            if 'bias' not  in name:
                param.data.mul_(masks[i].cuda())
                i += 1
        return True
    def apply_mask_plus_noise(self, masks,inverse_mask,noise,device):
        # masks = list(self.buffers())
        i = 0
        for (name, param) in self.named_parameters():
            if 'bias' not in name:
                desired_shape = param.data.shape
                thing = noise * torch.ones(desired_shape, device=device)
                thing = thing.mul(inverse_mask[i].cuda())
                param.data.mul_(masks[i].cuda()).add_(thing)
                i += 1
        return True

    def create_folder(self, path_to_file=""):
        prefix = f"{type(self).__name__}_model/"
        for name, param in self.named_parameters():
            if "bias" not in name:
                try:
                    os.makedirs(prefix + path_to_file + "/" + name.replace(".", "_")
                                + "/")
                except:
                    thing = prefix + path_to_file + "/" + name.replace(".", "_")
                    print(f"The directory {thing} already exist!!!!")
        return prefix + path_to_file

    def inspect_and_record_weigths(self, masks, path_to_file, epoch ,iteration):
        i = 0
        for name, param in self.named_parameters():
            if "bias" not in name:
                weigth = param.data[masks[i].type(torch.BoolTensor)].clone().cpu().detach().numpy().flatten()
                np.save(path_to_file + "/" + name.replace(".", "_") + f"/weigth_e{epoch}_i{iteration}", weigth)

                i += 1

    def inspect_and_record_gradients(self, masks, path_to_file, epoch ,iteration):
        i = 0
        for name, param in self.named_parameters():
            if "bias" not in name:
                weigth = param.grad.data[masks[i].type(torch.BoolTensor)].clone().cpu().detach().numpy().flatten()
                np.save(path_to_file + "/" + name.replace(".", "_") + f"/gradient_e{epoch}_i{iteration}", weigth)
                i += 1

    def partial_grad(self, data, target, loss_function):
        """
        Function to compute the grad
        args : data, target, loss_function
        return loss
        """
        outputs = self.forward(data)
        loss = loss_function(outputs, target)
        loss.backward()  # compute grad
        return loss

    def calculate_loss_grad(self, dataset, loss_function, n_samples):
        """
        Function to compute the full loss and the full gradient
        args : dataset, loss function and number of samples
        return : total loss and full grad norm
        """
        total_loss = 0.0

        for i_grad, data_grad in enumerate(dataset):
            inputs, labels = data_grad
            inputs, labels = Variable(inputs), Variable(labels)  # wrap data and target into variable
            total_loss += (1. / n_samples) * self.partial_grad(inputs, labels, loss_function).data[0]

        return total_loss

    def constricted_train(self, dataset, loss_function, n_epoch, learning_rate, momentum, iter_testloader,
                          file_name_sufix, inverse_mask,
                          restriction, optimizer="SGD", record=True):
        """
        Function to updated weights with a SVRG backpropagation
        args : dataset, loss function, number of epochs, learning rate
        return : total_loss_epoch, grad_norm_epoch
        """
        if optimizer == "SGD":
            if record:
                open(file_name_sufix + f"/test_restricted_training_value_{restriction}.txt", "w").close()
                open(file_name_sufix + f"/loss_restricted_training_value_{restriction}.txt", "w").close()
            velocity_buffer = {}
            dampening = 0
            nesterov = 0
            for epoch in range(n_epoch):
                running_loss = 0.0
                # previous_net_sgd = copy.deepcopy(self)  # update previous_net_sgd
                # previous_net_grad = copy.deepcopy(self)  # update previous_net_grad
                #
                # # Compute full grad
                # previous_net_grad.zero_grad()  # grad = 0
                #
                # Run over the dataset
                for i_data, data in enumerate(dataset):
                    inputs, labels = data
                    inputs, labels = inputs.cuda(), labels.cuda()  # wrap data and target into variable
                    # inputs.cuda()
                    # labels.cuda()
                    # Compute cur stoc grad
                    self.zero_grad()  # grad = 0
                    cur_loss = self.partial_grad(inputs, labels, loss_function)

                    # Backward
                    j = 0
                    for name, param in self.named_parameters():
                        d_p = param.grad.data
                        update_value = 0
                        if name not in velocity_buffer.keys():
                            buf = velocity_buffer[name] = torch.zeros_like(d_p)
                            buf.mul_(momentum).add_(d_p)

                        else:
                            buf = velocity_buffer[name]
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                        if "bias" not in name:
                            param.data.add_(-learning_rate, d_p)
                            with torch.no_grad():
                                param.data[inverse_mask[j].type(
                                    torch.BoolTensor)].clamp_(min=-restriction, max=restriction)
                            j += 1
                        else:
                            param.data.add_(-learning_rate, d_p)
                    # print statistics
                    running_loss += cur_loss.item()
                    if record:
                        with open(file_name_sufix + f"/loss_restricted_training_value_{restriction}.txt", "a") as f:
                            f.write(f"{cur_loss.item()}\n")
                        eval = evaluate_model(self, None, iter_testloader, partial=True)
                        with open(file_name_sufix + f"/test_restricted_training_value_{restriction}.txt", "a") as f:
                            f.write(f"{eval}\n")
                    if i_data % 200 == 199:  # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i_data + 1, running_loss / 2000))
                        running_loss = 0.0
        if optimizer == "kronecker":
            if record:
                open(file_name_sufix + f"/test_krone_restricted_training_value_{restriction}.txt", "w").close()
                open(file_name_sufix + f"/loss_krone_restricted_training_value_{restriction}.txt", "w").close()
            optimizer = KFACOptimizer(self)
            criterion = loss_function
            for epoch in range(n_epoch):
                running_loss = 0.0
                # Run over the dataset
                for i_data, data in enumerate(dataset):
                    inputs, labels = data
                    inputs, labels = inputs.cuda(), labels.cuda()  # wrap data and target into variable
                    # inputs.cuda()
                    # labels.cuda()
                    # Compute cur stoc grad
                    # self.zero_grad()  # grad = 0
                    # cur_loss = self.partial_grad(inputs, labels, loss_function)
                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    if optimizer.steps % optimizer.TCov == 0:
                        # compute true fisher
                        optimizer.acc_stats = True
                        with torch.no_grad():
                            sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1),
                                                          1).squeeze().cuda()
                        loss_sample = criterion(outputs, sampled_y)
                        loss_sample.backward(retain_graph=True)
                        optimizer.acc_stats = False
                        optimizer.zero_grad()  # clear the gradient for computing true-fisher.
                    loss.backward()
                    optimizer.step()
                    item = loss.item()
                    running_loss += item
                    if record:
                        with open(file_name_sufix + f"/loss_krone_restricted_training_value_{restriction}.txt",
                                  "a") as f:
                            f.write(f"{item}\n")
                        eval = evaluate_model(self, None, iter_testloader, partial=True)
                        with open(file_name_sufix + f"/test_krone_restricted_training_value_{restriction}.txt",
                                  "a") as f:
                            f.write(f"{eval}\n")
                    if i_data % 200 == 199:  # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i_data + 1, running_loss / 200))
                        running_loss = 0.0


def get_inverted_mask(model):
    if not hasattr(model, "buffers"):
        RuntimeError("Model must have been pruned in order to get inverted mask")
    params = list(model.buffers())
    mascara = []
    for param in params:
        temp = (param == 0).type(torch.double).detach().clone()
        mascara.append(temp)
    return mascara


def regularization(model, mask, device="cpu:0", distance=1, type="partial"):
    if type == "partial":
        suma = torch.tensor(0, device=device, dtype=torch.double)
        i = 0
        for (name, param) in model.named_parameters():
            if 'bias' not in name:
                suma = suma + torch.sum(torch.abs(param) * mask[i].cuda())
                i += 1
        return distance * suma
    if type == "whole":
        all_linear1_params = torch.cat([x.view(-1) for x in model.parameters()])
        return distance * torch.norm(all_linear1_params, 1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(model, dataset, iter_dataset, partial=False):
    if partial:
        assert iter_dataset is not None, "If you are using partial then you need to provide a iter of the " \
                                         "dataset"
        assert isinstance(iter_dataset, cycle), "Iter_dataset must be of type Cycle"
        model.eval()
        with torch.no_grad():
            data = next(iter_dataset)
            x, y = data
            y_pred = model(x.cuda())
            accuracy.update((y_pred, y.cuda()))
        thing = accuracy.compute()
        # print("Accuracy on one batch of testset: ", thing)
        accuracy.reset()
        return thing
    else:
        model.eval()
        with torch.no_grad():
            for data in dataset:
                x, y = data
                y_pred = model(x.cuda())
                accuracy.update((y_pred, y.cuda()))
        thing = accuracy.compute()
        print("Accuracy on testset: ", thing)
        accuracy.reset()
        return thing


def training(net, trainloader, testloader, optimizer, file_name_sufix, distance, mask, surname="", epochs=40,
             regularize=False, record_time=False, record_function_calls=False,record_weigths=False,
             record_gradients=False):
    import os
    try:
        os.mkdir(file_name_sufix)
    except:
        pass

    iter_testloader = cycle(testloader)
    if regularize:
        assert mask is not None, "If regularize=True then mask cannot be None"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # print labels
        import torch.optim as optim
        # net = Net()
        # net.to(device)
        # prune.global_unstructured(
        #     net.parameters_to_prune(),
        #     pruning_method=prune.L1Unstructured,
        #     amount=count_parameters(net) // 2,
        # )
        #
        # print(sum(torch.nn.utils.parameters_to_vector(net.buffers()) == 0))
        # mask = get_inverted_mask(net)
        criterion = nn.CrossEntropyLoss()
        net.cuda()
        net.ensure_device(device)
        open(file_name_sufix + f"/loss_regularized_training_lambda_{distance}.txt", "w").close()
        open(file_name_sufix + f"/test_regularized_training_lambda_{distance}.txt", "w").close()

        # epochs = 40
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss1 = criterion(outputs, labels)
                regu = regularization(net, mask, device, distance, type="whole")
                loss = loss1 + regu
                loss.backward()
                optimizer.step()

                # print statistics
                item_ = loss.item()
                running_loss += item_
                with open(file_name_sufix + f"/loss_regularized_training_{surname}_lambda_{distance}.txt", "a") as f:
                    f.write(f"{item_}\n")
                eval = evaluate_model(net, None, iter_testloader, partial=True)
                with open(file_name_sufix + f"/test_regularized_training_{surname}_lambda_{distance}.txt", "a") as f:
                    f.write(f"{eval}\n")
                if i % 200 == 199:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 200))
                    running_loss = 0.0

            evaluate_model(net, testloader, iter_dataset=None)
    if not regularize:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print labels
        import torch.optim as optim
        criterion = nn.CrossEntropyLoss()
        net.cuda()
        root_folder = ""
        if record_weigths or record_gradients:
            root_folder = net.create_folder(surname)
        net.ensure_device(device)
        if record_function_calls:
            open(file_name_sufix + "/function_call_" + surname + ".txt", "w").close()
        if record_time:
            open(file_name_sufix + "/time_" + surname + ".txt", "w").close()
        open(file_name_sufix + f"/test_training_{surname}.txt", "w").close()
        open(file_name_sufix + f"/loss_training_{surname}.txt", "w").close()
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                # zero the parameter gradients
                optimizer.zero_grad()
                if isinstance(optimizer, KFACOptimizer):
                    t0 = time.time_ns()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    if optimizer.steps % optimizer.TCov == 0:
                        # compute true fisher
                        optimizer.acc_stats = True
                        with torch.no_grad():
                            sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1),
                                                          1).squeeze().cuda()
                        loss_sample = criterion(outputs, sampled_y)
                        loss_sample.backward(retain_graph=True)
                        optimizer.acc_stats = False
                        optimizer.zero_grad()  # clear the gradient for computing true-fisher.
                    loss.backward()
                    optimizer.step()
                    t1 = time.time_ns()
                    if record_time:
                        with open(file_name_sufix + "/time_" + surname + ".txt", "a") as f:
                            f.write(str(t1 - t0) + "\n")
                    if record_function_calls:
                        with open(file_name_sufix + "/function_call_" + surname + ".txt", "a") as f:
                            f.write("2\n")

                    item = loss.item()
                    running_loss += item
                    with open(file_name_sufix + f"/loss_training_{surname}.txt", "a") as f:
                        f.write(f"{item}\n")
                    eval = evaluate_model(net, None, iter_testloader, partial=True)
                    with open(file_name_sufix + f"/test_training_{surname}.txt",
                              "a") as f:
                        f.write(f"{eval}\n")
                    if i % 200 == 199:  # print every 200 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 200))
                        if record_weigths:
                            net.inspect_and_record_weigths(mask, root_folder, epoch, i)
                        if record_gradients:
                            net.inspect_and_record_gradients(mask, root_folder, epoch, i)
                        running_loss = 0.0
                elif isinstance(optimizer, SAM):

                    t0 = time.time_ns()
                    # first forward-backward step
                    predictions = net(inputs)
                    loss = criterion(predictions, labels)
                    loss.backward()
                    item_ = loss.item()
                    optimizer.first_step(zero_grad=True)

                    # second forward-backward step
                    criterion(net(inputs), labels).backward()
                    optimizer.second_step(zero_grad=True)
                    t1 = time.time_ns()
                    if record_time:
                        with open(file_name_sufix + "/time_" + surname + ".txt", "a") as f:
                            f.write(str(t1 - t0) + "\n")
                    if record_function_calls:
                        with open(file_name_sufix + "/function_call_" + surname + ".txt", "a") as f:
                            f.write("2\n")
                    # print statistics
                    running_loss += item_
                    with open(file_name_sufix + f"/loss_training_{surname}.txt", "a") as f:
                        f.write(f"{item_}\n")
                    eval = evaluate_model(net, None, iter_testloader, partial=True)
                    with open(file_name_sufix + f"/test_training_{surname}.txt", "a") as f:
                        f.write(f"{eval}\n")
                    if i % 200 == 199:  # print every 200 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 200))
                        if record_weigths:
                            net.inspect_and_record_weigths(mask, root_folder, epoch, i)
                        if record_gradients:
                            net.inspect_and_record_gradients(mask, root_folder, epoch, i)
                        running_loss = 0.0
                else:
                    t0 = time.time_ns()
                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    t1 = time.time_ns()
                    if record_time:
                        with open(file_name_sufix + "/time_" + surname + ".txt", "a") as f:
                            f.write(str(t1 - t0) + "\n")
                    if record_function_calls:
                        with open(file_name_sufix + "/function_call_" + surname + ".txt", "a") as f:
                            f.write("1\n")
                    # print statistics
                    item_ = loss.item()
                    running_loss += item_
                    with open(file_name_sufix + f"/loss_training_{surname}.txt", "a") as f:
                        f.write(f"{item_}\n")
                    eval = evaluate_model(net, None, iter_testloader, partial=True)
                    with open(file_name_sufix + f"/test_training_{surname}.txt", "a") as f:
                        f.write(f"{eval}\n")
                    if i % 200 == 199:  # print every 200 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 200))
                        if record_weigths:
                            net.inspect_and_record_weigths(mask, root_folder, epoch, i)
                        if record_gradients:
                            net.inspect_and_record_gradients(mask, root_folder, epoch, i)
                        running_loss = 0.0
            evaluate_model(net, testloader, iter_dataset=None)


def read_regitered_weigths(root_directory="net_model", type="weight") -> dict:
    elements = [x[0] for x in os.walk(root_directory)]
    elements.pop(0)
    container = {}
    for name in elements:
        for file in glob.glob(f"{name}/{type}*"):
            print(file)
            epoch = int(file[file.rfind("_e"):file.rfind("_i")].replace("_e",""))
            iteration = int(file[file.rfind("_i",-15):file.rfind(".n",-15)].replace("_i",""))

            # s1 = f"epoch {epoch}"
            # s2 = f"iteration {iteration}"
            real_name = name.replace(root_directory,"").replace("\\","")
            if real_name not in container.keys():
                container[real_name] = {epoch: {iteration: np.load(file)}}
            else:
                container[real_name].update({epoch: {iteration: np.load(file)}})
    return container


def test_against_original(dataset):
    original_weight = []
    original_model = Net()
    original_model.load_state_dict(torch.load("model_trained_vanila"))
    for p in original_model.parameters():
        original_weight.extend(p.detach().numpy().flatten())
    original_model.cuda()
    evaluate_model(original_model, dataset, None)

    distances = [1, 10, 100, 200, 300, 400]
    import pandas as pd

    dataframe = pd.DataFrame({"original": original_weight})
    for l in distances:

        current_net = Net()
        current_net.load_state_dict(torch.load(f"model_trained_fullreg_{l}"))
        current_net.cuda()
        evaluate_model(current_net, dataset, None)
        current_weigths = []
        for p in current_net.parameters():
            current_weigths.extend(p.cpu().detach().numpy().flatten())
        from scipy import stats
        dataframe[f"model {l}"] = current_weigths

        stast_, p_value = stats.ttest_ind(original_weight, current_weigths)
        print(f"Testing Model number {l} against original")
        print(f"T-Statistic: {stast_}")
        print(f"P-value: {p_value}")
    dataframe.boxplot()
    plt.show()


def load_CIFAR10(datapath, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=datapath, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=datapath, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    batch_size = 32
    cluster = True
    datapath = ""
    if cluster:
        datapath = "/nobackup/sclaam/data"
    else:
        datapath = "./data"
    trainloader, testloader = load_CIFAR10(datapath, batch_size)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # net = Net()
    import torch.optim as optim

    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # training(net,trainloader,testloader,optimizer,"traces",mask = None,distance=0,epochs=10)
    # torch.save(net.state_dict(), "model_trained_vanila")
    # print("Performance before pruning")
    # evaluate_model(net,testloader,iter_dataset=None)
    # prune.global_unstructured(
    #             net.parameters_to_prune(),
    #             pruning_method=prune.L1Unstructured,
    #             amount=count_parameters(net) // 2,
    #         )
    # print("Performance after pruning\n")
    # evaluate_model(net,testloader,iter_dataset=None)
    # torch.save(net.state_dict(), "model_trained_vanila_pruned")
    # model2 = ModelPytorch()
    # model2.cuda()
    # model = Net()
    # prune.global_unstructured(
    #     model.parameters_to_prune(),
    #     pruning_method=prune.L1Unstructured,
    #     amount=count_parameters(net) // 2,
    # )
    # model.load_state_dict(torch.load("model_trained_vanila_pruned"))
    # model.cuda()
    # evaluate_model(model2,testloader,None)
    # evaluate_model(model, testloader, None)
    # mask = get_inverted_mask(model)
    # k = 0
    # buffers = list(model.buffers())
    # for (name,module)in model2.named_modules():
    #     if not isinstance(module,ModelPytorch) and not isinstance(module,nn.MaxPool2d):
    #         module.set_masks(buffers[k])
    #         k += 1
    # Tainig with mask from the beginning: ###########################################################################
    # optimizer = optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)
    # training(model2,trainloader,testloader,optimizer,"traces",epochs=10,distance=0,mask=None)
    # torch.save(model2.state_dict(), f"model_trained_pruned_scratch")
    #

    # Trainnig with distance ##########################################################################################
    # distances = [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # for l in distances:
    #     new_net = Net()
    #     optimizer = optim.SGD(new_net.parameters(), lr=0.001, momentum=0.9)
    #     training(new_net, trainloader, testloader, optimizer, "traces", mask=mask, regularize=True, distance=l,
    #              epochs=10)
    #     torch.save(net.state_dict(), f"model_trained_lambda_{l}")

    # Training with restriction ########################################################################################
    # distances = [80,10,5,2,1,0.1,0.01,0.001,0.0001,0.00001,0]
    # distances = [0.001,0.0001]
    # for restriction in distances:
    #     net = Net()
    #     loss = nn.CrossEntropyLoss()
    #     net.cuda()
    #     net.constricted_train(trainloader, loss, 10, 0.001, 0.9, cycle(testloader), "traces", mask, restriction)
    #     torch.save(net.state_dict(), f"model_restricted_trained_{restriction}")
    #     evaluate_model(net, testloader, iter_dataset=None)

    # Train different regularized models and compare their weigths statistically ######################################
    # distances = [1, 10, 100, 200, 300, 400, 500] #600, 700, 800, 900, 1000]
    # for l in distances:
    #     new_net = Net()
    #     optimizer = optim.SGD(new_net.parameters(), lr=0.001, momentum=0.9)
    #     training(new_net, trainloader, testloader, optimizer, "traces", mask=mask, regularize=True, distance=l,
    #              epochs=10)
    #
    #     torch.save(new_net.state_dict(), f"model_trained_fullreg_{l}")
    # test_against_original(testloader)

    # # Train with Kronecker-factored optimizer
    # #########################################################################
    # k = 0
    # buffers = list(model.buffers())
    # for (name,module)in model2.named_modules():
    #     if not isinstance(module,ModelPytorch) and not isinstance(module,nn.MaxPool2d):
    #         module.set_masks(buffers[k])
    #         k += 1
    # #Tainig with mask from the beginning:
    # optimizer = KFACOptimizer(model2,lr=0.001,momentum=0.5)
    # training(model2,trainloader,testloader,optimizer,"traces",epochs=10,distance=0,mask=None)
    # torch.save(model2.state_dict(), f"model_trained_pruned_scratch_krone")

    ####################################################################################################################
    # Train small network with Kronecker and big Network with SGD and Kronecker
    # model = SmallNet()
    # optimizer = KFACOptimizer(model,lr=0.001,momentum=0.5)
    # training(model,trainloader,testloader,optimizer,"traces",surname="KFAC_small" ,epochs=10,distance=0,mask=None)
    # torch.save(model.state_dict(), f"model_small_trained_KFAC")
    #
    # model = Net()
    # optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
    # training(model,trainloader,testloader,optimizer,"traces",surname="SGD_big",epochs=10,distance=0,mask=None)
    # torch.save(model.state_dict(), f"model_big_trained_SGD")
    #
    # model = Net()
    # optimizer = KFACOptimizer(model,lr=0.001,momentum=0.5)
    # training(model,trainloader,testloader,optimizer,"traces",surname="KFAC_big" ,epochs=10,distance=0,mask=None)
    # torch.save(model.state_dict(), f"model_big_trained_KAFC")
    #
    # model = SmallNet()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # training(model, trainloader, testloader, optimizer, "traces", surname="SGD_small", epochs=10, distance=0, mask=None)
    ###################################################################################################################
    # Train  small networks with less convolutional layers and trained with KFAC . Also trained bigger network with SGD
    # path_colab = "/content/drive/MyDrive/Colab Notebooks/Extra-dimension-role/"
    #
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"Device used {device}")
    # small_model = NewSmallNet()
    # optimizer = SAM(small_model.parameters(), optim.SGD, lr=0.01, momentum=0.9)
    # training(small_model, trainloader, testloader, optimizer, "traces", surname="SAM_conv_small", epochs=10, distance=0,
    #          mask=None, record_function_calls=True, record_time=True)
    # torch.save(small_model.state_dict(), f"model_small_trained_SAM")
    #
    # small_model = NewSmallNet()
    #
    # optimizer = KFACOptimizer(small_model, lr=0.001, momentum=0.5)
    # training(small_model, trainloader, testloader, optimizer, "traces", surname="KFAC_conv_small", epochs=10,
    #          distance=0,
    #          mask=None, record_function_calls=True, record_time=True)
    # torch.save(small_model.state_dict(), f"model_small_trained_KFAC")
    # #
    # big_model = NewNet()
    # optimizer = optim.SGD(big_model.parameters(), lr=0.001, momentum=0.9)
    # training(big_model, trainloader, testloader, optimizer, "traces", surname="SGD_conv_big", epochs=10, distance=0,
    #          mask=None, record_function_calls=True, record_time=True)
    # torch.save(big_model.state_dict(), f"model_big_trained_SGD")
    #
    # small_model = NewSmallNet()
    # optimizer = optim.SGD(small_model.parameters(), lr=0.001, momentum=0.9)
    # training(small_model, trainloader, testloader, optimizer, "traces", surname="SGD_conv_small", epochs=10,
    #          distance=0,
    #          mask=None, record_function_calls=True, record_time=True)
    # torch.save(small_model.state_dict(), f"model_small_trained_SGD")
    # #
    #
    # big_model = NewNet()
    # optimizer = KFACOptimizer(big_model, lr=0.001, momentum=0.5)
    # training(big_model, trainloader, testloader, optimizer, "traces", surname="KFAC_conv_big", epochs=10, distance=0,
    #          mask=None, record_function_calls=True, record_time=True)
    # torch.save(big_model.state_dict(), f"model_big_trained_KFAC")

    """
     The experiments in this next section are to find what happens if the training begings with a mask but no 
     restriction  is imposed in the weights outside the mask. Will the weights grow  and then come back to begin 
     small? 
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ref_net = Net()
    model_dict = ref_net.state_dict()
    # prune.global_unstructured(
    #     net.parameters_to_prune(),
    #     pruning_method=prune.L1Unstructured,
    #     amount=count_parameters(ref_net) // 2,
    # )
    ref_net.load_my_state_dict(torch.load("model_trained_vanila_pruned"))
    ref_net.cuda()
    evaluate_model(ref_net,testloader,None)
    mask = get_inverted_mask(ref_net)
    net = Net()
    net.cuda()
    net.apply_mask_plus_noise(list(ref_net.buffers()),mask,0,device)
    # folder = net.create_folder("test")
    # net.inspect_and_record_weigths(mask, folder, 1, 1)
    # net.inspect_and_record_weigths(mask, folder, 1, 200)
    # net.inspect_and_record_weigths(mask, folder, 2, 1)
    # net.inspect_and_record_weigths(mask, folder, 2, 200)
    # x, y = next(iter(trainloader))
    # pred = net(x.cuda())
    # loss = nn.CrossEntropyLoss()
    # temp = loss(pred,y.cuda())
    # temp.backward()
    # net.inspect_and_record_gradients(mask, folder, 1, 1)
    # net.inspect_and_record_gradients(mask, folder, 1, 200)
    # net.inspect_and_record_gradients(mask, folder, 2, 1)
    # net.inspect_and_record_gradients(mask, folder, 2, 200)
    # a = read_regitered_weigths(folder, type="weigth")
    # b = read_regitered_weigths(folder, type="gradient")
    # things = []
    from sam import  SAM
    optimizer = SAM(net.parameters(),optim.SGD,lr=0.01,momentum=0.9)
    training(net, trainloader, testloader, optimizer,"traces_trash",0,mask,"SAM_pruned_test",epochs=10,
             record_weigths=True,record_gradients=True)


