import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt

batch_size = 4

transform = torchvision.transforms.ToTensor()

trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# print(images.shape)

image_size = 28 * 28

class DigitRecognize(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(image_size, 64)
        self.fc2 = nn.Linear(64, 10)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, image_size)
        return self.fc2(self.activation(self.fc1(x)))


model = DigitRecognize()
criterion = nn.CrossEntropyLoss()
optimizer2 = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 3

for epoch in range(epochs):
    correct = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        correct += torch.sum(pred.argmax(dim=1) == labels).item()

    print(f"Epoch {epoch+1} complete with {correct} correct!")

