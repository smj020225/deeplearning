import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import model
# 定义VGG13模型

# 设置训练参数
learning_rate = 0.001
batch_size = 64
epochs = 10

# 准备CIFAR-10数据集
transform = transforms.Compose([
    transforms.Resize(240,240),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.VGG13().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 开始训练
from tqdm import tqdm

# 定义VGG13模型

# 设置训练参数
learning_rate = 0.001
batch_size = 256
epochs = 40

# 准备CIFAR-10数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.VGG13().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


import matplotlib.pyplot as plt
from tqdm import tqdm


train_loss_history = []
train_accuracy_history = []
# 开始训练
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct_predictions = 0.0

    # Wrap your train_loader with tqdm
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 计算准确度
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()

    average_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / len(train_dataset) * 100
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}, Accuracy: {accuracy:.2f}%')
    model_save_path = '/content/drive/MyDrive/vgg13.pth'
    torch.save(model.state_dict(), model_save_path)
    train_loss_history.append(average_loss)
    train_accuracy_history.append(average_loss)

    # 保存训练好的模型
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracy_history, label='Train Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/training_history.png')
plt.show()

