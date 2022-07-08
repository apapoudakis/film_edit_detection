from torch.utils.data import DataLoader
import torch


def train_loop(model, train_data, test_data, batch_size, num_epochs, device, optimizer, criterion):

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    model = model.to(device)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_epoch_losses = []
        test_epoch_losses = []
        for it, data in enumerate(train_loader):
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

        with torch.no_grad():
            for test_it, data in enumerate(test_loader):
                test_x, test_labels = data
                test_x = test_x.to(device)
                test_labels = test_labels.to(device)

                test_outputs = model(test_x)
                loss = criterion(test_outputs, test_labels)
                test_epoch_losses.append(loss.item())

        train_losses.append(sum(train_epoch_losses)/len(train_epoch_losses))
        test_losses.append(sum(test_epoch_losses)/len(test_epoch_losses))

        print(f"Epoch: {epoch+1} Train Loss: {train_losses[epoch]} Test Loss: {test_losses[epoch]}")
