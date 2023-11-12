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
model.train_model(train_loader, 2, device)

print("Testing the model...")
model.test_model(test_loader)



# DNN verification



from z3 import *

solver = Solver()

s = [784,200,100,60,30]
L = []

layers = []
layers_after_relu = []

y = []
x = []
b = []
W = []

n = 5

def init_vars():
    x = [ [ Real(f"x_{j}_{i}") for j in range(s[i]) ] for i in range(n) ]
    y = [ [ Real(f"y_{j}_{i}") for j in range(s[i]) ] for i in range(n) ]
    b = [ [ Real(f"b_{j}_{i}") for j in range(s[i]) ] for i in range(n) ]
    W = [ [ [Real(f"w_{j}_{i}_{k}") for k in range(s[i-1])] for j in range(s[i]) ] for i in range(1,n) ]

init_vars()

def get_contraint(N,x_star, L,j):
    def eq_layer(x,y):
        And([x[i] == y[i] for i in range(s[0])])


    def c_in(x):
        eq_layer(x,layers[0])

    def matrice_product(x,y):
        temp = 0
        for i in range(len(x)):
            temp = temp + x[i] * y[i]

    def c(i):
        And([ y[i][j] == matrice_product(W[i-1][j],x[i - 1]) + b[i][j] for j in range(s[i]) ])


    def c_prime(i):
        And( [  And(Implies(y[i][j] > 0, x[i][j] == y[i][j]), Implies(y[i][j] <= 0, x[i][j] == 0) )  for j in range(s[i]) ] )

    def c_out(L,j):
        And([  x[n][k] <= x[n][j]  for k in range(s[-1]) if k != j ])

    temp = [And(c(i),c_prime(i)) for i in range(1,n+1)]
    temp.append(c_in(x_star))
    temp.append(c_out(L,j))
    And(temp)



def Max(x,y):
    If(x > y, x, y)

def distance(x,y):
    temp = 0
    for i in range(len(x)):
        temp = Max(temp,Abs(x[i] - y[i]))
    temp


def find_epsilon() :
    epsilon = -0.1
    is_sat = false

    solver.add(get_contrain(N,x,L_star,j))

    while not(is_sat):
        print(f"test: {epsilon}")
        epsilon += 0.1
        solver.add(distance(x_star,x) < epsilon)
        is_sat = solver.sat() == sat
        if not(is_sat) :
            solver.pop()
    print(f"find epsilon : {epsilon}")
