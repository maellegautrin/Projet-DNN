#!/usr/bin/env python3



# Definition de l'ia
# Merci à Felix


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 60)
        self.fc4 = nn.Linear(60, 30)
        self.fc5 = nn.Linear(30, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.softmax(x)
        return x


    def test_model(self, dataloader):
        criterion = nn.CrossEntropyLoss()
        self.to(device)
        self.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()


        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total

        print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    def train_model(self, dataloader, epochs, device):
        learning_rate=1e-3
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.9995)
        self.to(device)

        for epoch in tqdm(range(epochs)):
            self.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                outputs = self(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

            scheduler.step()
            avg_loss = total_loss / len(dataloader)
            accuracy = 100 * correct / total

            print(f"Epoch [{epoch + 1}/{epochs}]: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")



# get the data
print("Importing data...")
batch_size = 100
transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
        ])
dataset1 = datasets.MNIST('./MNIST', train=True, download=False,
                    transform=transform)
dataset2 = datasets.MNIST('./MNIST', train=False,
                    transform=transform)


train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=False)


# get the model

print("Training the model...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using gpu: {torch.cuda.is_available()}')
model = LinearNet().to(device)
#model.train_model(train_loader, 2, device)

print("Testing the model...")
model.test_model(test_loader)



# DNN verification



from z3 import *

solver = Solver()


# Init vars

W = []
b = []
s = []
for layer in model.children():
    W.append(layer.state_dict()['weight'])
    b.append(layer.state_dict()['bias'])
    s.append(layer.state_dict()['weight'].size()[1])
s.append(10)
n = len(s)
x = [ [ Real(f"x_{i}_{j}") for j in range(s[i]) ] for i in range(n) ]
y = [ [ Real(f"y_{i}_{j}") for j in range(s[i]) ] for i in range(n) ]

print(f"layers detected : {s}")



# Define contraints

def get_contraint(x_var, label):
    def eq_layer(x,y):
        for i in range(s[0]):
            solver.add(x[i] == y[i])

    def c_in(x_var):
        eq_layer(x_var,x[0])

    def matrice_product(x,y):
        temp = 0
        for i in range(len(x)):
            temp = temp + float(x[i]) * y[i]
        return temp

    def c(i):
        for j in range(s[i+1]):
            solver.add( y[i][j] == matrice_product(W[i][j],x[i]) + float(b[i][j]))


    def c_prime(i):
        for j in range(s[i]):
            solver.add(Implies(y[i][j] > 0, x[i][j] == y[i][j]), Implies(y[i][j] <= 0, x[i][j] == 0) )

    def c_out(label):
        print(label)
        print(len(x[n-1]))
        for k in range(s[-1]) :
            if k != label :
                solver.add(x[n-1][k] <= x[n-1][label])

    for i in range(0,n-1):
        c(i)
        c_prime(i)
    c_in(x_var)
    c_out(label)



def Max(x,y):
    return If(x > y, x, y)

def distance(x,y):
    temp = 0
    for i in range(len(x)):
        temp = Max(temp,Abs(float(x[i]) - y[i]))
    return temp


def find_epsilon(label,x_star) :
    epsilon = -0.1
    is_sat = False



    x_var = [Real(f"xvar_{i}") for i in range(s[0])]

    print("generate contraints")
    get_contraint(x_var,label)
    solver.push()

    print("find epsilon")
    while not(is_sat):
        epsilon += 0.1
        print(f"test: {epsilon}")
        solver.add(distance(x_star,x_var) < epsilon)
        is_sat = solver.check() == sat
        if not(is_sat) :
            solver.pop()
    print(f"find epsilon : {epsilon}")
    return epsilon

def find_second_label(entry):
    output = model(entry)
    first = 0
    second = 0
    firsti = 0
    secondi = 0
    i = 0
    for l in output:
        if l > first:
            second = first
            secondi = firsti
            first = l
            first = i
        elif l > second :
            second = l
            secondi = i
        i += 1
    return secondi


second_label = find_second_label(dataset2[0][0])
epsilon = find_epsilon(second_label, dataset2[0][0])
