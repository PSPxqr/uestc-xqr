
import random
import torch
#build base
def synthetic_data(w, b, num_examples):
    x=torch.normal(mean=0,std=1,size=(num_examples,len(w)))
    y=torch.matmul(x,w)+b
    y+=torch.normal(mean=0,std=0.03,size=y.shape)
    return x,y.reshape((-1,1))
true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=synthetic_data(true_w,true_b,1000)

#read little data
def data_iter(banch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,banch_size):
        banch_indices=torch.tensor(indices[i:min(i+banch_size, num_examples)])
    yield features[banch_indices],labels[banch_indices]

banch_size=4

if __name__ == '__main__':
    for x,y in data_iter(banch_size,features,labels):
        print(x,'\n',y)
        break

#linear
w=torch.normal(mean=0,std=1,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

def linear(x,w,b):
    return torch.matmul(x,w)+b

def squared_loss(y_hat,y):
    return torch.pow(y_hat-y.reshape(y_hat.shape),2)*0.5

def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()

#process
lr=0.5
num_epochs=10
net=linear
loss=squared_loss

for epoch in range(num_epochs):
    for x,y in data_iter(banch_size,features,labels):
        l=loss(net(x,w,b),y)
        l.sum().backward()
        sgd([w, b], lr, banch_size)
    with torch.no_grad():
        train_loss=loss(net(features,w,b),labels)
        print(f"epoch{epoch+1},loss{float(train_loss.mean()): f}")


