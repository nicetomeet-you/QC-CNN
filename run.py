import torch
import torch.nn as nn
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from app.load_data import MyCSVDatasetReader as CSVDataset
from models.classical import Net
from models.single_encoding import Net,PreConvNet
#from models.multi_encoding import Net
#from models.hybrid_layer import Net
#from models.inception import Net
#from models.multi_noisy import Net
from app.train import train_network,train_qcnn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
import os
# load the dataset
dataset = CSVDataset('./datasets/mnist_179_1200.csv')

parser = argparse.ArgumentParser()
parser.add_argument("--Log", default="D:\Pycharm\RangeNet\QC-CNN\Log", help="the path where model saved")
parser.add_argument("--preModel", default=" ", help="the path where pre-trained model is")
FLAGS = parser.parse_args()
# output location/file names
# outdir = 'results_255_tr_mnist358'
# file_prefix = 'mnist_358'


# load the device
device = torch.device('cpu')

# define model
net = PreConvNet()
net_single = Net()
# net.to(device)
# # 训练数据和测试数据的下载
# 训练数据和测试数据的下载
trainDataset = torchvision.datasets.MNIST( # torchvision可以实现数据集的训练集和测试集的下载
    root="./data", # 下载数据，并且存放在data文件夹中
    train=True, # train用于指定在数据集下载完成后需要载入哪部分数据，如果设置为True，则说明载入的是该数据集的训练集部分；如果设置为False，则说明载入的是该数据集的测试集部分。
    transform=transforms.ToTensor(), # 数据的标准化等操作都在transforms中，此处是转换
    download=True # 瞎子啊过程中如果中断，或者下载完成之后再次运行，则会出现报错
)

testDataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
batch_size_train = 64
batch_size_test = 10000
train_loader = torch.utils.data.DataLoader(trainDataset,batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(testDataset,batch_size=batch_size_test, shuffle=True)

criterion = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.Adagrad(net.parameters(), lr = 0.5) # optimizer

epochs = 10
bs = 30

# train_id, val_id = train_test_split(list(range(len(dataset))), test_size = 0.2, random_state = 0)
# train_set = Subset(dataset, train_id)
# val_set = Subset(dataset, val_id)
# train_network(net = net_single, train_set = train_set, val_set = val_set, device = device, 
# epochs = epochs, bs = bs, optimizer = optimizer, criterion = criterion)  # outdir = outdir, file_prefix = file_prefix)
if os.path.exists(FLAGS.preModel):
    net.load_state_dict(torch.load(FLAGS.preModel))
train_qcnn(net = net, train_loader=train_loader, val_loader= test_loader, device = device, 
epochs = epochs, optimizer = optimizer, criterion = criterion)  # outdir = outdir, file_prefix = file_prefix)

# train_network(net = net, train_set = train_set, val_set = val_set, device = device, 
# epochs = epochs, bs = bs, optimizer = optimizer, criterion = criterion)  # outdir = outdir, file_prefix = file_prefix)
