import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sam import SAM


class BigFullyC(nn.Module):

    def __init__(self, image_W, image_H, classes):
        super(BigFullyC, self).__init__()
        self.fc1 = nn.Linear(image_W * image_H * 3, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        y = self.fc5(x)
        return y

    def parameters_to_prune(self):
        layers = []
        for module in self.modules():
            if not isinstance(module, BigFullyC):
                layers.append(module)
        weights = ["weight"] * len(layers)
        return list(zip(layers, weights))

    def ensure_device(self, device):
        for module in self.modules():
            module.to(device)

    def apply_mask(self, masks):
        # masks = list(self.buffers())
        i = 0
        for (name, param) in self.named_parameters():
            if 'bias' not in name:
                param.data.mul_(masks[i].cuda())
                i += 1
        return True


class SmallFullyC(nn.Module):
    def __init__(self, image_W, image_H, classes):
        super(SmallFullyC, self).__init__()
        self.fc1 = nn.Linear(image_W * image_H * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)

    def forward(self, x) -> torch.Tensor:
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = self.fc1(x)
        x = self.fc2(x)
        y = self.fc3(x)
        return y

    def parameters_to_prune(self):
        layers = []
        for module in self.modules():
            if not isinstance(module, BigFullyC):
                layers.append(module)
        weights = ["weight"] * len(layers)
        return list(zip(layers, weights))

    def ensure_device(self, device):
        for module in self.modules():
            module.to(device)

    def apply_mask(self, masks):
        # masks = list(self.buffers())
        i = 0
        for (name, param) in self.named_parameters():
            if 'bias' not in name:
                param.data.mul_(masks[i].cuda())
                i += 1
        return True


class SmallConv(nn.Module):
    def __init__(self, classes):
        super(SmallConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.fc1 = nn.Linear(1024, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        y = self.fc1(x)
        return y

    def parameters_to_prune(self):
        layers = []
        for module in self.modules():
            if not isinstance(module, BigFullyC):
                layers.append(module)
        weights = ["weight"] * len(layers)
        return list(zip(layers, weights))

    def ensure_device(self, device):
        for module in self.modules():
            module.to(device)

    def apply_mask(self, masks):
        # masks = list(self.buffers())
        i = 0
        for (name, param) in self.named_parameters():
            if 'bias' not in name:
                param.data.mul_(masks[i].cuda())
                i += 1
        return True


class BigConv(nn.Module):
    def __init__(self, classes):
        super(BigConv, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        self.conv4 = nn.Conv2d(256, 512, 3, padding=(1, 1))
        self.conv5 = nn.Conv2d(512, 1024, 3)

        self.fc1 = nn.Linear(4096, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(x), (2, 2))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(F.relu(self.conv5(x)), (2, 2))
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        y = self.fc1(x)
        return y

    def parameters_to_prune(self):
        layers = []
        for module in self.modules():
            if not isinstance(module, BigConv):
                layers.append(module)
        weights = ["weight"] * len(layers)
        return list(zip(layers, weights))

    def ensure_device(self, device):
        for module in self.modules():
            module.to(device)

    def apply_mask(self, masks):
        # masks = list(self.buffers())
        i = 0
        for (name, param) in self.named_parameters():
            if 'bias' not in name:
                param.data.mul_(masks[i].cuda())
                i += 1
        return True


if __name__ == '__main__':
    from main import training, load_CIFAR10
    from KFAC_Pytorch.optimizers.kfac import KFACOptimizer

    # datapath = "/nobackup/sclaam/data"
    # path_colab = "/content/drive/MyDrive/Colab Notebooks/Extra-dimension-role/"
    trainloader, testloader = load_CIFAR10("./data", 32)
    # TRAINING PURE FULLY CONNECTED

    # Small  FC net SGD optimizer
    small = SmallFullyC(32, 32, 10)
    optimizer = optim.SGD(small.parameters(), lr=0.001, momentum=0.9)
    training(small, trainloader, testloader, optimizer, "pure_experiments", 0, None, "FC_small_SGD", epochs=10,
             regularize=False, record_time=True, record_function_calls=True)
    torch.save(small.state_dict(), f"FC_small_SGD")
    # Big FC net SGD optimizer
    big = BigFullyC(32, 32, 10)
    optimizer = optim.SGD(big.parameters(), lr=0.001, momentum=0.9)
    training(big, trainloader, testloader, optimizer, "pure_experiments", 0, None, "FC_big_SGD", epochs=10,
             regularize=False, record_time=True, record_function_calls=True)
    torch.save(big.state_dict(), f"FC_big_SGD")

    # Small  FC net SAM optimizer
    small = SmallFullyC(32, 32, 10)
    optimizer = SAM(small.parameters(), optim.SGD, lr=0.01, momentum=0.9)
    training(small, trainloader, testloader, optimizer, "pure_experiments", 0, None, "FC_small_SAM", epochs=10,
             regularize=False, record_time=True, record_function_calls=True)
    torch.save(small.state_dict(), f"FC_small_SAM")

    # Big FC net SAM optimizer
    big = BigFullyC(32, 32, 10)
    optimizer = SAM(big.parameters(), optim.SGD, lr=0.01, momentum=0.9)
    training(big, trainloader, testloader, optimizer, "pure_experiments", 0, None, "FC_big_SAM", epochs=10,
             regularize=False, record_time=True, record_function_calls=True)
    torch.save(big.state_dict(), f"FC_big_SAM")

    # Small model KFAC optimizer
    small = SmallFullyC(32, 32, 10)
    optimizer = KFACOptimizer(small, lr=0.001, momentum=0.5)
    training(small, trainloader, testloader, optimizer, "pure_experiments", distance=0, mask=None,
             surname="FC_small_KFAC", epochs=10, record_time=True, record_function_calls=True)
    torch.save(small.state_dict(), f"FC_small_KFAC")

    # Big model KFAC optimizer
    big = BigFullyC(32, 32, 10)
    optimizer = KFACOptimizer(big, lr=0.001, momentum=0.5)
    training(big, trainloader, testloader, optimizer, "pure_experiments", distance=0, mask=None, surname="FC_big_KFAC",
             epochs=10, record_time=True, record_function_calls=True)
    torch.save(big.state_dict(), f"FC_big_KFAC")
    ############################################################################################################
    # NOW THE SAME EXACT EXPERIMENTS BUT WITH CONVOLUTIONAL NETWORK ONLY.
    # #
    #
    # # Small  CONV net SGD optimizer
    # print("SMALL CONV NET SGD")
    # small = SmallConv(10)
    # # big_temp = BigConv(10)
    # # x,y = next(iter(trainloader))
    # # small.forward(x)
    # # big_temp.forward(x)
    # optimizer = optim.SGD(small.parameters(), lr=0.001, momentum=0.9)
    # training(small, trainloader, testloader, optimizer, "pure_experiments", 0, None, "CONV_small_SGD", epochs=10,
    #          regularize=False, record_time=True, record_function_calls=True)
    # torch.save(small.state_dict(), f"CONV_small_SGD")
    #
    # # Big CONV net SGD optimizer
    # print("BIG CONV NET SGD")
    # big = BigConv(10)
    # optimizer = optim.SGD(big.parameters(), lr=0.001, momentum=0.9)
    # training(big, trainloader, testloader, optimizer, "pure_experiments", 0, None, "CONV_big_SGD", epochs=10,
    #          regularize=False, record_time=True, record_function_calls=True)
    # torch.save(big.state_dict(), f"CONV_big_SGD")
    #
    # # Small  CONV net SAM optimizer
    # print("SMALL CONV NET SAM")
    # small = SmallConv(10)
    # optimizer = SAM(small.parameters(), optim.SGD, lr=0.01, momentum=0.9)
    # training(small, trainloader, testloader, optimizer, "pure_experiments", 0, None, "CONV_small_SAM", epochs=10,
    #          regularize=False, record_time=True, record_function_calls=True)
    # torch.save(small.state_dict(), f"CONV_small_SAM")
    #
    # # Big CONV net SAM optimizer
    # print("BIG CONV NET SAM")
    # big = BigConv(10)
    # optimizer = SAM(big.parameters(), optim.SGD, lr=0.01, momentum=0.9)
    # training(big, trainloader, testloader, optimizer, "pure_experiments", 0, None, "CONV_big_SAM", epochs=10,
    #          regularize=False, record_time=True, record_function_calls=True)
    # torch.save(big.state_dict(), f"CONV_big_SAM")
    #
    # # Small CONV KFAC optimizer
    # print("SMALL CONV NET KFAC")
    # small = SmallConv(10)
    # optimizer = KFACOptimizer(small, lr=0.001, momentum=0.5)
    # training(small, trainloader, testloader, optimizer, "pure_experiments", surname="CONV_small_KFAC", epochs=10,
    #          distance=0,
    #          mask=None, record_function_calls=True, record_time=True)
    # torch.save(small.state_dict(), f"CONV_small_KFAC")
    #
    # # Big model KFAC optimizer
    # print("BIG CONV NET KFAC")
    # big = BigConv(10)
    # optimizer = KFACOptimizer(big, lr=0.001, momentum=0.5)
    # training(big, trainloader, testloader, optimizer, "pure_experiments", surname="CONV_big_KFAC", epochs=10,
    #          distance=0,
    #          mask=None, record_function_calls=True, record_time=True)
    # torch.save(big.state_dict(), f"CONV_big_KFAC")
