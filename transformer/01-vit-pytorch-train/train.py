"""Train cifar10 with ViT.
- Author: Hyeonwoo Jeong
- Contact: qhsh9713@gmail.com
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from model.vit import ViT
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Training settings
EPOCHS = 10

# Training settings
cifar10_classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def get_dataset(
    batch_size: int = 32, num_workers: int = 2
) -> tuple[DataLoader, DataLoader]:
    """Get train / test dataset."""
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return trainloader, testloader


def train(epoch, model, criterion, optimizer, trainloader) -> float:
    """Train model for one epoch."""
    print(f"Train Epoch: {epoch}")
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(trainloader.dataset)} "
                f"({100. * batch_idx / len(trainloader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
    return train_loss / (batch_idx + 1)


def test(model, criterion, testloader) -> tuple[float, float]:
    """Test model."""
    global best_acc
    print("Test")
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(testloader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(
                    f"Test: [{batch_idx * len(data)}/{len(testloader.dataset)} "
                    f"({100. * batch_idx / len(testloader):.0f}%)]\tLoss: {loss.item():.6f}"
                )

    # Save checkpoint. .pt (Only save the best model)
    acc = 100.0 * correct / total
    print(f"Accuracy: {acc:.3f}")
    if acc > best_acc:
        print("Saving..")
        state = {
            "model": model.state_dict(),
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(state, "./checkpoint/best_ckpt.pth")
        best_acc = acc
    return test_loss, acc


if __name__ == "__main__":
    ########################################################################
    # Dataset
    ########################################################################
    trainloader, testloader = get_dataset(batch_size=32, num_workers=2)

    ########################################################################
    # Model
    ########################################################################
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=10,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    ).to(device)

    ########################################################################
    # Training hyperparameters setting
    ########################################################################
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    ########################################################################
    # Training
    ########################################################################
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)
        train_loss = train(epoch, model, criterion, optimizer, trainloader)
        test_loss, test_acc = test(model, criterion, testloader)
        print(
            f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )
        scheduler.step()

    print("Finished Training")
