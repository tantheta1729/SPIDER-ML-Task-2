import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. DATA PREPARATION SETTINGS
# ==========================================
os.makedirs('./data', exist_ok=True)

# ==========================================
# 2. ACCELERATED DATA LOADING PIPELINES
# ==========================================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15), 
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# ==========================================
# 3. RESNET STRUCTURAL MODULES
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=0.2)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        identity = self.shortcut(identity)
        out += identity
        out = self.relu(out)
        return out

class CustomResNet(nn.Module):
    def __init__(self, block, num_blocks_per_stage, num_classes=10):
        super(CustomResNet, self).__init__()
        self.in_channels = 64

        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(block, 64, num_blocks_per_stage[0], stride=1)
        self.stage2 = self._make_stage(block, 128, num_blocks_per_stage[1], stride=2)
        self.stage3 = self._make_stage(block, 256, num_blocks_per_stage[2], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_stage(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.prep(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# ==========================================
# 4. ENGINE SETUP FUNCTIONS
# ==========================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CustomResNet(ResidualBlock, num_blocks_per_stage=[2, 2, 2], num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.15, momentum=0.9, weight_decay=1e-3)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 25], gamma=0.1)

# ==========================================
# 5. CORE EXECUTION ENGINES
# ==========================================
def train_one_epoch(epoch, trainloader):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return train_loss / (batch_idx + 1), 100. * correct / total

def test_model(testloader):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return test_loss / (batch_idx + 1), 100. * correct / total

# ==========================================
# 6. PROTECTED WINDOWS MAIN RUNTIME ENTRYWAY
# ==========================================
if __name__ == '__main__':
    print("Setting up Windows localized DataLoaders...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    print(f"\nLaunching Tracked Training Profile on {device}...\n" + "-"*85)
    
    # Start tracking global script time
    script_start_time = time.time()

    for epoch in range(1, 31):
        # Start tracking individual epoch time
        epoch_start_time = time.time()
        
        current_lr = optimizer.param_groups[0]['lr']
        tr_loss, tr_acc = train_one_epoch(epoch, trainloader)
        te_loss, te_acc = test_model(testloader)

        train_losses.append(tr_loss)
        train_accuracies.append(tr_acc)
        test_losses.append(te_loss)
        test_accuracies.append(te_acc)

        # Calculating duration of the current epoch
        epoch_duration = time.time() - epoch_start_time
        
        print(f"Epoch {epoch:02d} (LR: {current_lr:.4f}) | Train Acc: {tr_acc:.2f}% | Test Acc: {te_acc:.2f}% | Time: {epoch_duration:.2f}s")
        scheduler.step()

    # Calculating overall script time
    total_script_duration = time.time() - script_start_time
    total_minutes = int(total_script_duration // 60)
    total_seconds = total_script_duration % 60

    torch.save(model.state_dict(), 'custom_resnet_cifar10.pth')
    print("\n" + "="*85)
    print(f"[SUCCESS] Total Run Time: {total_minutes}m {total_seconds:.2f}s")
    print("Model weights safely exported to 'custom_resnet_cifar10.pth'!")
    print("="*85)

    # ==========================================
    # 7. PLOTTING METRICS
    # ==========================================

    print("\n" + "="*40 + "\nGENERATING SUBMISSION DELIVERABLES\n" + "="*40)
    epochs_range = range(1, len(train_accuracies) + 1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy', color='blue', linewidth=2)
    plt.plot(epochs_range, test_accuracies, label='Validation Accuracy', color='orange', linewidth=2)
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='LR Drop (to 0.015)')
    plt.axvline(x=20, color='purple', linestyle='--', alpha=0.7, label='LR Drop (to 0.0015)')
    plt.axvline(x=25, color='green', linestyle='--', alpha=0.7, label='LR Drop (to 0.00015)')
    plt.title('30-Epoch Accuracy History Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(epochs_range, test_losses, label='Validation Loss', color='orange', linewidth=2)
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.7)
    plt.axvline(x=20, color='purple', linestyle='--', alpha=0.7)
    plt.axvline(x=25, color='green', linestyle='--', alpha=0.7)
    plt.title('30-Epoch Loss History Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())

    print("\n" + "="*40 + "\nCLASSIFICATION PRECISION REPORT\n" + "="*40)
    print(classification_report(all_targets, all_preds, target_names=classes))

    print("\n" + "="*40 + "\nCONFUSION MATRIX PLOT\n" + "="*40)
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Final Validation Confusion Matrix Heatmap')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()