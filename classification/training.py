import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torchvision.models as models
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from torchvision.models import ResNet18_Weights


# Define transformations for training and validation datasets
transform = transforms.Compose([
    transforms.Resize((224, 224)),# ResNet18 expects 224x224 input images
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(10),    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3952, 0.3817, 0.3761], std=[0.2371, 0.2313, 0.2301]),
])

# Load datasets
train_dataset = datasets.ImageFolder(root='clsdata/train', transform=transform)
val_dataset = datasets.ImageFolder(root='clsdata/val', transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)





# Load the pretrained ResNet18 model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Variables to track metrics
train_acc_history = []
val_acc_history = []
train_f1_history = []
val_f1_history = []

# Training loop
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Track accuracy and F1 score
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    train_acc_history.append(epoch_acc)
    train_f1_history.append(epoch_f1)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {epoch_acc}, F1 Score: {epoch_f1}')
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_labels = []
    val_preds = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())
    
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='weighted')
    
    val_acc_history.append(val_acc)
    val_f1_history.append(val_f1)
    
    print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {val_acc}, F1 Score: {val_f1}')

torch.save(model.state_dict(), 'model.pth')
print("Model saved")