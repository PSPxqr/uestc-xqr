import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader

from linear.linear import synthetic_data as d2l

true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=d2l(true_w,true_b,1000)

def load_array(data_array, batch_size,is_training=True):
    dataest=data.TensorDataset(*data_array)
    return DataLoader(dataest,batch_size,shuffle=is_training)

batch_size=10
data_iter=load_array(features,batch_size)

next(iter(data_iter))

from torch import nn
net=nn.Sequential(nn.Linear(2,1))

net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

loss=nn.MSELoss()
trainer=torch.optim.SGD(net.parameters(),lr=0.03)

num_epoch=3
for epoch in range(num_epoch):
    for x,y in data_iter:
        l=loss(net(x),y)
        l.backward()
        trainer.step()
    l=loss(net(features),labels)
    print(f"epoch{epoch+1},loss{l:f}")

w=net[0].weight.data
b=net[0].bias.data
print(f"误差：{true_w-w.reshape(true_w.shape)}")
print(f"b误差: {b-true_b}")