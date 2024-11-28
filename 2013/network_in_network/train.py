import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train_and_test_model(model, model_name, dataset_filepath='../../data',
                         batch_size=32, learning_rate=0.01,
                         momentum=0.9, weight_decay=0.001, factor=0.1,
                         patience=10, threshold=0.01, num_workers=4):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR10(root=dataset_filepath, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = datasets.CIFAR10(root=dataset_filepath, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    model = model.to(device)

    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # ReduceLROnPlateau:
    # lower learning rate when a metric has stopped improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience, threshold_mode="rel", threshold=threshold)

    best_loss = float("inf")
    best_epoch = -1


    test_accuracies = []

    def train(epoch):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criteria(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch}] Step [{i + 1}/{len(train_loader)}] Loss: {loss.item()}')

    def test(epoch, best_loss, best_epoch):
        model.eval()
        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criteria(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()

        avg_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)

        print(f'Epoch [{epoch}] Test Loss: {avg_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')


        test_accuracies.append(accuracy)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f'model_best_{model_name}.pth')

        return best_loss, best_epoch, accuracy


    epoch = 0
    while scheduler.get_last_lr()[0]  >= 0.001 * learning_rate:
        train(epoch)
        best_loss, best_epoch, accuracy = test(epoch, best_loss, best_epoch)
        scheduler.step(accuracy)
        epoch += 1

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoch + 1), test_accuracies, marker='o', linestyle='-', color='b')
    plt.title('Test Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.ylim(0, 100)


    plt.savefig(f'test_accuracy_plot_{model_name}.png')
    plt.close()

    return best_loss, best_epoch


if __name__ == '__main__':

    from models.network_in_network import NIN
    nin = NIN(3, 10)

    # set hyperparameters
    batch_size = 128
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 0.001

    best_loss, best_epoch = train_and_test_model(nin, 'nin', batch_size=batch_size, learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay)