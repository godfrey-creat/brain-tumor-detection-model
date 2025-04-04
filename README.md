# Brain Tumor Detection Model

## Overview
This project implements a Convolutional Neural Network (CNN) for brain tumor detection using MRI and CT scan images. The model is trained using PyTorch and can classify images into different categories based on the presence of a brain tumor.

## Dataset
The dataset consists of MRI and CT scan images organized into labeled folders. Images are preprocessed and transformed before training the model.

### Sample Dataset Images


![image](https://github.com/user-attachments/assets/9a432af9-e1b1-4f90-8eb6-74aeb2aa0c0a)


## Installation
Ensure you have Python installed and the following dependencies:
```bash
pip install torch torchvision matplotlib pathlib
```

## Project Structure
```
brain-tumor-detection-model/
│-- dataset/  # Contains MRI and CT scan images
│-- models/model.ipynb   # Saved model checkpoints
│-- scripts/app.py  # Training and evaluation scripts
│-- README.md # Project documentation
```

## Code Breakdown

### Data Preprocessing
The dataset is loaded using PyTorch's `ImageFolder` and transformed:
```python
import pathlib
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

dataset_path = pathlib.Path("C:/Users/PC/Downloads/mri-ct/Dataset")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=str(dataset_path), transform=transform)
```

### Splitting Data
The dataset is split into 80% training and 20% validation:
```python
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
```

### Data Loaders
```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

## Model Architecture
The CNN model consists of three convolutional layers followed by fully connected layers.
```python
import torch
import torch.nn as nn
import torch.optim as optim

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

## Model Training
The training loop uses cross-entropy loss and Adam optimizer:
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    history["train_loss"].append(running_loss / len(train_loader))
    history["train_acc"].append(correct / total)
```

### Training Progress Graph



![image](https://github.com/user-attachments/assets/4c266750-f366-4c69-b216-0d636cf282f1)

## Model Evaluation
```python
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
print(f'Validation Accuracy: {correct / total:.4f}')
```

## Saving the Model
```python
torch.save(model.state_dict(), "brain_tumor_model.pth")
print("Model saved successfully!")
```

## Usage
To train the model:
```bash
python app.py
```
To evaluate the model:
```bash
python evaluate.py
```

## Future Improvements
- Fine-tuning with pre-trained models (e.g., ResNet, EfficientNet)
- Experimenting with different optimizers and learning rates
- Extending the dataset for better generalization

## Contributors
- **Godfrey Otieno** - Developer

## License
This project is licensed under the MIT License.


