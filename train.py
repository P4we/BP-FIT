import hydra
import torch
from torch import nn
from torchvision import datasets, transforms


class DigitRecognize(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 10)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.fc2(self.activation(self.fc1(x)))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    config = cfg.training
    transform = transforms.ToTensor()

    train_set = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config.train_batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=config.test_batch_size, shuffle=False
    )

    model = DigitRecognize()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    epochs = config.epochs

    for epoch in range(epochs):
        print("starting epoch:", epoch)
        for image, label in train_loader:
            optimizer.zero_grad()
            pred = model(image)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
        print("epoch", epoch, "finished")

    correct = 0
    total = 0

    print("starting evaluation")
    with torch.no_grad():
        for image, label in test_loader:
            preds = model(image).argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

    print("\nfinal accuracy:", correct / total * 100, "%")


if __name__ == "__main__":
    main()
