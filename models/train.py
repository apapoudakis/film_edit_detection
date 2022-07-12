from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from evaluation.metrics import accuracy


def train_loop(model, train_data, test_data, batch_size, num_epochs, device):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    model = model.to(device)

    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        train_epoch_losses = []
        test_epoch_losses = []
        num_correct_train = 0
        num_correct_test = 0

        for it, data in enumerate(tqdm(train_loader)):
            x, labels = data
            x = x.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_epoch_losses.append(loss.item())
            num_correct_train = (outputs == labels).float().sum()
        with torch.no_grad():
            for test_it, data in enumerate(tqdm(test_loader)):
                test_x, test_labels = data
                test_x = test_x.to(device)
                test_labels = test_labels.to(device)

                test_outputs = model(test_x)
                test_loss = criterion(test_outputs, test_labels)
                test_epoch_losses.append(test_loss.item())
                num_correct_test = (test_outputs == test_labels).float().sum()

        train_losses.append(sum(train_epoch_losses)/len(train_epoch_losses))
        test_losses.append(sum(test_epoch_losses)/len(test_epoch_losses))
        train_accuracy.append(num_correct_train/len(train_data))
        test_accuracy.append(num_correct_test/len(test_data))

        print(f"Epoch: {epoch+1} Train Loss: {train_losses[epoch]} Test Loss: {test_losses[epoch]}")
        print(f"Train Accuracy: {train_accuracy[epoch]}  Test Accuracy: {test_accuracy[epoch]}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.show()
