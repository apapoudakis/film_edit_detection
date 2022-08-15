from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from utils import checkpoints


def train_loop(model, train_data, test_data, batch_size, num_epochs, device, checkpoint=None):

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MultiMarginLoss()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    start_epoch = 0

    if checkpoint is not None:
        model_state_dict, optimizer_state_dict, epoch, loss = checkpoints.load(checkpoint)
        start_epoch = epoch
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)

    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []

    for epoch in range(start_epoch, num_epochs):
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
            _, pred = torch.max(outputs.data, 1)
            num_correct_train += (pred == labels).sum().item()
        with torch.no_grad():
            for test_it, data in enumerate(tqdm(test_loader)):
                test_x, test_labels = data
                test_x = test_x.to(device)
                test_labels = test_labels.to(device)

                test_outputs = model(test_x)
                test_loss = criterion(test_outputs, test_labels)
                test_epoch_losses.append(test_loss.item())
                _, test_pred = torch.max(test_outputs.data, 1)
                num_correct_test += (test_pred == test_labels).sum().item()

        scheduler.step()
        train_losses.append(sum(train_epoch_losses)/len(train_epoch_losses))
        test_losses.append(sum(test_epoch_losses)/len(test_epoch_losses))
        train_accuracy.append(num_correct_train/len(train_data))
        test_accuracy.append(num_correct_test/len(test_data))

        print(f"Epoch: {epoch+1} Train Loss: {train_losses[epoch]} Test Loss: {test_losses[epoch]}")
        print(f"Train Accuracy: {num_correct_train/len(train_data)}  Test Accuracy: {num_correct_test/len(test_data)}")

        checkpoints.save(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                         epoch=epoch, loss=loss.item(), out_path="deepSBD_checkpoint" + str(epoch) + ".pt")

    torch.save(model.state_dict(), "model.pt")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.show()
