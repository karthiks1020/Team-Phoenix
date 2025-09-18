
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import os
import json
import time

def train_model():
    """
    Main function to handle the complete model training and saving pipeline.
    """
    print("🚀 Starting the model training pipeline...")

    # --- 1. Load Data and Define Transforms ---
    data_dir = './data/raw/'
    if not os.path.exists(data_dir) or not any(os.scandir(data_dir)):
        print(f"❌ Error: Data directory '{data_dir}' is empty or does not exist.")
        print("Please ensure your data is structured correctly (e.g., ./data/raw/pottery/image.jpg).")
        return

    # Define transformations for training and validation sets
    # ImageNet mean and std
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]),
    }

    print("📂 Loading image data from disk...")
    # Use ImageFolder to load the dataset
    full_dataset = datasets.ImageFolder(data_dir)
    
    # Automatically detect class names
    class_names = full_dataset.classes
    num_classes = len(class_names)
    class_to_idx = full_dataset.class_to_idx
    
    print(f"✅ Found {num_classes} classes: {', '.join(class_names)}")

    # --- 2. Split Dataset ---
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * 0.20)
    train_size = dataset_size - val_size
    
    print(f"Splitting dataset: {train_size} for training, {val_size} for validation.")
    
    # Perform the 80/20 split
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply the correct transforms to each split
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']

    # --- 3. Create DataLoaders ---
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    # --- 4. Define the Model ---
    print("🛠️  Setting up the model (ResNet-18)...")
    # Load a pre-trained ResNet-18 model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all layers in the pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully-connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    print("✅ Model classifier layer replaced.")

    # --- 5. Train the Model ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Training will use device: {device}")

    criterion = nn.CrossEntropyLoss()
    # Only optimize the parameters of the new classifier layer
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    num_epochs = 25
    
    since = time.time()
    best_acc = 0.0

    print(f"\n🔥 Starting training for {num_epochs} epochs...\n")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                print(f"Train Loss: {epoch_loss:.4f}", end=' | ')
            else:
                print(f"Val Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.4f}")
                # Deep copy the model if it's the best so far
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    # Save the best model state
                    best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"🏆 Best validation Accuracy: {best_acc:.4f}")

    # Load best model weights before saving
    model.load_state_dict(best_model_wts)

    # --- 6. Save the Final Artifacts ---
    models_dir = './models/'
    os.makedirs(models_dir, exist_ok=True)
    print(f"\n💾 Saving model and class indices to '{models_dir}'...")

    # 1. Save the model state dictionary
    model_path = os.path.join(models_dir, 'handicraft_cnn.pth')
    torch.save(model.state_dict(), model_path)
    print(f"   - Model saved to: {model_path}")

    # 2. Save the class mapping
    indices_path = os.path.join(models_dir, 'class_indices.json')
    with open(indices_path, 'w') as f:
        json.dump(class_to_idx, f, indent=4)
    print(f"   - Class indices saved to: {indices_path}")
    
    print("\n🎉 Pipeline finished successfully!")

if __name__ == '__main__':
    train_model()
