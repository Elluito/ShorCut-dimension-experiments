import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sam import SAM

class BigFullyC(nn.Module):

    def __init__(self,image_W,image_H,classes):
        super(BigFullyC, self).__init__()
        self.fc1 = nn.Linear(image_W*image_H,2048)
        self.fc2 = nn.Linear(2048,1024)
        self.fc3 = nn.Linear(1024,512)
        self.fc4 = nn.Linear(512,256)
        self.fc5 = nn.Linear(256,classes)

    def forward(self, x:torch.Tensor, **kwargs: Any) -> T_co:
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        y = self.fc5(x)
        return y
class SmallFullyC(nn.Module):
    def __init__(self,image_W,image_H,classes):
        super(SmallFullyC, self).__init__()
        self.fc1 = nn.Linear(image_W*image_H,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,classes)
    def forward(self,x)->torch.Tensor:
        x = x.view(-1,int(x.nelement()/x.shape[0]))
        x = self.fc1(x)
        x = self.fc2(x)
        y = self.fc3(x)
        return y
class SmallConv(nn.Module):
    def __init__(self):
        super(SmallConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.fc1   = nn.Linear(2304,classes)
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x),(2,2)))
        x = F.max_pool2d(F.relu(self.conv2(x),(2,2)))
        x = F.max_pool2d(F.relu(self.conv3(x),(2,2)))
        x = x.view(-1,int(x.nelement()/x.shape[0]))
        y = self.fc1(x)
        return y

class BigConv(nn.Module):
    def __init__(self,image_W,image_H,classes):
        super(BigConv, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        self.conv4 = nn.Conv2d(256, 512, 3,padding = (1,1))
        self.conv5 = nn.Conv2d(512, 1024, 3)

        self.fc1   = nn.Linear(9*1024,classes)
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv3(x) ),(2, 2))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(F.relu(self.conv5(x)),(2,2))
        x = x.view(-1,int(x.nelement()/x.shape[0]))
        y = self.fc1(x)
        return y
if __name__ == '__main__':
    from main import training,load_CIFAR10
    from KFAC_Pytorch.optimizers.kfac import KFACOptimizer
    datapath = "/nobackup/sclaam/data"
    treinloader,testloader = load_CIFAR10(datapath)
    # TRAINING PURE FULLY CONNECTED
    # Small  FC net SGD optimizer
    small = SmallFullyC(32,32,10)
    optimizer = optim.SGD(small.parameters(),lr=0.001,momentum=0.9)
    training(small,trainloader,testloader,optimizer,"pure_experiments",0,None,"FC_small_SGD",epochs=10,
             regularize=False,record_time=True,record_function_calls=True)
    torch.save(small.state_dict(), f"FC_small_SGD")
    # Big FC net SGD optimizer
    big = BigFullyC(32,32,10)
    optimizer = optim.SGD(big.parameters(), lr=0.001, momentum=0.9)
    training(big, trainloader, testloader, optimizer, "pure_experiments", 0, None, "FC_big_SGD", epochs=10,
         regularize=False, record_time=True, record_function_calls=True)
    torch.save(big.state_dict(), f"FC_big_SGD")

    # Small  FC net SAM optimizer
    small = SmallFullyC(32, 32, 10)
    optimizer = SAM(small.parameters(),optim.SGD ,lr=0.01, momentum=0.9)
    training(small, trainloader, testloader, optimizer, "pure_experiments", 0, None, "FC_small_SAM", epochs=10,
             regularize=False, record_time=True, record_function_calls=True)
    torch.save(small.state_dict(), f"FC_small_SAM")
    # Big FC net SAM optimizer
    big = BigFullyC(32, 32, 10)
    optimizer = SAM(big.parameters(),optim.SGD ,lr=0.01, momentum=0.9)
    training(big, trainloader, testloader, optimizer, "pure_experiments", 0, None, "FC_big_SAM", epochs=10,
             regularize=False, record_time=True, record_function_calls=True)
    torch.save(big.state_dict(), f"FC_big_SAM")
    # Small model KFAC optimizer
    small = SmallFullyC(32,32,10)
    optimizer = KFACOptimizer(small, lr=0.001, momentum=0.5)
    training(small, trainloader, testloader, optimizer, "pure_experiments", surname="FC_small_KFAC", epochs=10,
             distance=0,
             mask=None, record_function_calls=True, record_time=True)
    torch.save(small.state_dict(), f"FC_small_KFAC")

    # Big model KFAC optimizer
    big = BigFullyC(32,32,10)
    optimizer = KFACOptimizer(big, lr=0.001, momentum=0.5)
    training(big, trainloader, testloader, optimizer, "pure_experiments", surname="FC_big_KFAC", epochs=10,
             distance=0,
             mask=None, record_function_calls=True, record_time=True)
    torch.save(big.state_dict(), f"FC_big_KFAC")
