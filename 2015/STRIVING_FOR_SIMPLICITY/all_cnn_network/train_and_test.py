import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train_and_test_model(model_class, model_name, dataset_filepath='../../../data',
                         batch_size=32, epochs=350, learning_rate=0.01,
                         momentum=0.9, weight_decay=0.001, gamma=0.1,
                         milestones=[200, 250, 300]):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(root=dataset_filepath, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.CIFAR10(root=dataset_filepath, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model_c = model_class().to(device)

    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_c.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    best_loss = float("inf")
    best_epoch = -1


    test_accuracies = []

    def train(epoch):
        model_c.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model_c(images)
            loss = criteria(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}] Step [{i + 1}/{len(train_loader)}] Loss: {loss.item()}')

    def test(epoch, best_loss, best_epoch):
        model_c.eval()
        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model_c(images)
                loss = criteria(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()

        avg_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)

        print(f'Epoch [{epoch}/{epochs}] Test Loss: {avg_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')


        test_accuracies.append(accuracy)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save(model_c.state_dict(), f'model_c_best_{model_name}.pth')

        return best_loss, best_epoch


    for epoch in range(epochs):
        train(epoch)
        scheduler.step()
        best_loss, best_epoch = test(epoch, best_loss, best_epoch)


    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), test_accuracies, marker='o', linestyle='-', color='b')
    plt.title('Test Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.ylim(0, 100)


    plt.savefig(f'test_accuracy_plot_{model_name}.png')
    plt.close() 

    return best_loss, best_epoch


from models.All_CNN_C import AllCNNC
train_and_test_model(AllCNNC, 'All_CNN_C',weight_decay=1e-6)


# from models.base_model_c import BaseModelC
# train_and_test_model(BaseModelC, 'Base_Model_C')