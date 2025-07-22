import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from data.aptos_dataset import EyeDataset
from models.simple_cnn import SimpleCNN
import torch.nn.functional as F

# Load model
model = SimpleCNN(num_classes=5)
model.load_state_dict(torch.load("global_model.pth"))  # Save this after training
model.eval()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load test data
test_dataset = EyeDataset("D:\\federated learning\\APTOS\\test.csv", "D:\\federated learning\\APTOS\\test_images", transform)
test_loader = DataLoader(test_dataset, batch_size=16)

# Evaluate
loss, correct, total = 0.0, 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        loss += F.cross_entropy(preds, y, reduction='sum').item()
        correct += (preds.argmax(1) == y).sum().item()
        total += y.size(0)

print(f"Test Loss: {loss / total:.4f}")
print(f"Test Accuracy: {correct / total:.4f}")
