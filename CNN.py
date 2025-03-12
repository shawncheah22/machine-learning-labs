#%%
import torch 
from torchvision import transforms, datasets
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

#input size of image is 28x28

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)

        self.fc1 = nn.Linear(7*7*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        return x
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 7*7*64)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)
Epochs = 3 

for epoch in range(Epochs):
    for data in trainset:
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 1, 28, 28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

correct = 0 
total = 0 

with torch.no_grad():
    for data in testset:
        X, y = data
        output = net.forward(X)
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3))