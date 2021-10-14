import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import torch.nn as nn
from torch.nn import Flatten
from itertools import cycle
import torchvision
import torchvision.transforms as transforms
from ignite.metrics import Accuracy
import torch

# from KFAC_Pytorch.optimizers.kfac import KFACOptimizer

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
accuracy = Accuracy()
from definitions import LinearMasked, Conv2dMasked
from definitions import MaskedModule
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from KFAC_Pytorch.optimizers import KFACOptimizer


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def custom_model_tensorflow(compile_=True, Wdecay=0):
    model = tf.keras.Sequential([
        tf.keras.Input([32, 32, 3]),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(Wdecay)),
        # tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPool2D(),
        # tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(Wdecay)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(516, activation='relu', kernel_regularizer=l2(Wdecay)),
        tf.keras.layers.Dense(10, kernel_regularizer=l2(Wdecay))])
    if compile_:
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001, momentum=0.9),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    return model


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
        weights = ["weight"]*len(layers)
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(4608, 516)
        self.fc2 = nn.Linear(516, 10)

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


def get_inverted_mask(model):
    if not hasattr(model, "buffers"):
        RuntimeError("Model must have been pruned in order to get inverted mask")
    params = list(model.buffers())
    # suma = torch.tensor(0,device=model.device)
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
             regularize=False):
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
        net.ensure_device(device)
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
                    item = loss.item()
                    running_loss += item
                    with open(file_name_sufix + f"/loss_training_{surname}.txt", "a") as f:
                        f.write(f"{item}\n")
                    eval = evaluate_model(net, None, iter_testloader, partial=True)
                    with open(file_name_sufix + f"/test_training_{surname}.txt",
                              "a") as f:
                        f.write(f"{eval}\n")
                    if i % 200 == 199:  # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 200))
                        running_loss = 0.0
                else:
                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
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

                        running_loss = 0.0
            evaluate_model(net, testloader, iter_dataset=None)


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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 32

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)

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
    small_model = NewSmallNet()
    # l = NewNet()
    # x,y = next(iter(trainloader))
    optimizer = KFACOptimizer(small_model,lr=0.001,momentum=0.5)
    training(small_model,trainloader,testloader,optimizer,"traces",surname="KFAC_conv_small",epochs=10,distance=0,
             mask=None)
    # torch.save(model.state_dict(), f"model_big_trained_KAFC")



    big_model = NewNet()
    optimizer = optim.SGD(big_model.parameters(), lr=0.001, momentum=0.9)
    training(big_model, trainloader, testloader, optimizer, "traces", surname="SGD_conv_big", epochs=10, distance=0,
             mask=None)

    big_model = NewNet()
    optimizer = KFACOptimizer(big_model,lr=0.001,momentum=0.5)
    training(big_model,trainloader,testloader,optimizer,"traces",surname="KFAC_conv_big" ,epochs=10,distance=0,
             mask=None)

