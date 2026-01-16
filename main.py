import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50
IMAGE_SIZE = 224

# Dataset Class with better error handling
class FaceShapeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Filter out corrupted images during initialization
        self.valid_indices = []
        print("Validating images...")
        for idx in tqdm(range(len(image_paths)), desc="Checking images"):
            try:
                img = Image.open(image_paths[idx])
                img.verify()  # Verify it's a valid image
                self.valid_indices.append(idx)
            except Exception as e:
                print(f"Skipping corrupted image: {image_paths[idx]}")
        
        print(f"Valid images: {len(self.valid_indices)}/{len(image_paths)}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img_path = self.image_paths[real_idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[real_idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            blank_image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='black')
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image, self.labels[real_idx]

# Enhanced data transformations
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Improved Transfer Learning Model using ResNet50
class FaceShapeResNet(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(FaceShapeResNet, self).__init__()
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Replace the final fully connected layer with better architecture
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

# Load dataset function with improved filtering
def load_dataset(data_dir, split='Train'):
    image_paths = []
    labels = []
    class_names = []
    
    split_path = Path(data_dir) / split
    
    # Filter out non-directory items and system files
    valid_dirs = []
    for class_dir in sorted(split_path.iterdir()):
        if class_dir.is_dir() and class_dir.name not in ['desktop.ini', '.DS_Store', '__MACOSX', 'Thumbs.db']:
            valid_dirs.append(class_dir)
    
    for idx, class_dir in enumerate(valid_dirs):
        class_names.append(class_dir.name)
        image_count = 0
        
        for img_path in class_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                # Skip system files
                if img_path.name.lower() not in ['desktop.ini', '.ds_store', 'thumbs.db']:
                    image_paths.append(str(img_path))
                    labels.append(idx)
                    image_count += 1
        
        print(f"  {class_dir.name}: {image_count} images")
    
    print(f"\nTotal: {len(image_paths)} images in {split} set across {len(class_names)} classes")
    print(f"Classes: {class_names}")
    
    return image_paths, labels, class_names

# Training function with mixed precision and gradient clipping
def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training for faster computation
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# Main training loop
def main():
    DATA_DIR = './face-shape-dataset'
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory '{DATA_DIR}' not found!")
        return
    
    # Load data
    print("\n" + "="*60)
    print("Loading Training Data...")
    print("="*60)
    train_paths, train_labels, class_names = load_dataset(DATA_DIR, 'Train')
    
    print("\n" + "="*60)
    print("Loading Validation Data...")
    print("="*60)
    val_paths, val_labels, val_class_names = load_dataset(DATA_DIR, 'Val')
    
    if class_names != val_class_names:
        print(f"WARNING: Train and Val have different classes!")
        print(f"Train classes: {class_names}")
        print(f"Val classes: {val_class_names}")
        return
    
    num_classes = len(class_names)
    print(f"\n{'='*60}")
    print(f"Dataset Summary")
    print(f"{'='*60}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Classes: {class_names}")
    
    # Create datasets (this will filter corrupted images)
    print("\n" + "="*60)
    print("Creating Training Dataset...")
    print("="*60)
    train_dataset = FaceShapeDataset(train_paths, train_labels, train_transform)
    
    print("\n" + "="*60)
    print("Creating Validation Dataset...")
    print("="*60)
    val_dataset = FaceShapeDataset(val_paths, val_labels, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=0, pin_memory=True)
    
    # Initialize model with transfer learning
    print("\n" + "="*60)
    print("Initializing Model...")
    print("="*60)
    print("Loading pre-trained ResNet50...")
    model = FaceShapeResNet(num_classes=num_classes, pretrained=True).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-7
    )
    
    # Mixed precision scaler for faster training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 15
    
    print("\n" + "="*60)
    print("Starting Training with Transfer Learning...")
    print("="*60)
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Early Stop Patience: {early_stop_patience}")
    print("="*60 + "\n")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.7f}")
        print("-" * 60)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
                'train_acc': train_acc,
                'val_loss': val_loss
            }, 'best_face_shape_model.pth')
            print(f"  ✓ NEW BEST! Saved model (improvement: +{improvement:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epoch(s)")
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            print(f"{'='*60}")
            break
    
    # Plot training history
    print("\nGenerating training visualizations...")
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc', marker='o', linewidth=2)
    plt.plot(history['val_acc'], label='Val Acc', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    epochs = range(1, len(history['val_acc']) + 1)
    colors = ['green' if acc == best_val_acc else 'skyblue' for acc in history['val_acc']]
    plt.bar(epochs, history['val_acc'], alpha=0.7, color=colors)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.title('Validation Accuracy per Epoch', fontsize=14, fontweight='bold')
    plt.axhline(y=best_val_acc, color='red', linestyle='--', linewidth=2, 
                label=f'Best: {best_val_acc:.2f}%')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Training history saved to 'training_history.png'")
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Total Epochs Run: {len(history['val_acc'])}")
    print(f"Model saved as: best_face_shape_model.pth")
    print(f"{'='*60}\n")

# Test the model on test set
def test_model(model_path, data_dir, device):
    """Evaluate model on test set and show detailed metrics"""
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    print("\n" + "="*60)
    print("TESTING MODEL ON TEST SET")
    print("="*60)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    
    print(f"Loaded model from epoch {checkpoint['epoch']+1}")
    print(f"Model's validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    model = FaceShapeResNet(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    print("\n" + "="*60)
    print("Loading Test Data...")
    print("="*60)
    test_paths, test_labels, _ = load_dataset(data_dir, 'Test')
    
    print("\nCreating Test Dataset...")
    test_dataset = FaceShapeDataset(test_paths, test_labels, val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=0, pin_memory=True)
    
    print(f"\n{'='*60}")
    print(f"Test samples (after filtering): {len(test_dataset)}")
    print(f"{'='*60}\n")
    
    # Make predictions
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Running inference on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    # Calculate accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    test_acc = 100 * np.mean(all_preds == all_labels)
    
    print(f"\n{'='*60}")
    print(f"TEST ACCURACY: {test_acc:.2f}%")
    print(f"{'='*60}")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 12})
    plt.title(f'Confusion Matrix - Test Accuracy: {test_acc:.2f}%', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✓ Confusion matrix saved to 'confusion_matrix.png'")
    
    # Per-class accuracy
    print(f"\n{'='*60}")
    print("PER-CLASS ACCURACY:")
    print(f"{'='*60}")
    for i, class_name in enumerate(class_names):
        class_mask = all_labels == i
        class_correct = np.sum((all_preds[class_mask] == all_labels[class_mask]))
        class_total = np.sum(class_mask)
        class_acc = 100 * class_correct / class_total if class_total > 0 else 0
        print(f"  {class_name:12s}: {class_acc:6.2f}% ({class_correct}/{class_total} correct)")
    
    # Average confidence for correct/incorrect predictions
    correct_mask = all_preds == all_labels
    correct_confidences = [all_probs[i][all_preds[i]] for i in range(len(all_preds)) if correct_mask[i]]
    incorrect_confidences = [all_probs[i][all_preds[i]] for i in range(len(all_preds)) if not correct_mask[i]]
    
    print(f"\n{'='*60}")
    print("CONFIDENCE ANALYSIS:")
    print(f"{'='*60}")
    if correct_confidences:
        print(f"  Average confidence (correct):   {np.mean(correct_confidences)*100:.2f}%")
    if incorrect_confidences:
        print(f"  Average confidence (incorrect): {np.mean(incorrect_confidences)*100:.2f}%")
    print(f"{'='*60}\n")
    
    return test_acc

# Predict on a single image with visualization
def predict_and_visualize(model_path, image_path, device):
    """Predict face shape and visualize the result"""
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    
    model = FaceShapeResNet(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and predict
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    
    # Get prediction
    predicted_idx = torch.argmax(probabilities).item()
    predicted_class = class_names[predicted_idx]
    confidence = probabilities[predicted_idx].item()
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Show image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(f'Predicted: {predicted_class}\nConfidence: {confidence:.1%}', 
                  fontsize=16, fontweight='bold', pad=10)
    
    # Show probabilities
    colors = ['green' if i == predicted_idx else 'skyblue' for i in range(num_classes)]
    bars = ax2.barh(class_names, probabilities.cpu().numpy() * 100, color=colors)
    ax2.set_xlabel('Probability (%)', fontsize=13)
    ax2.set_title('Class Probabilities', fontsize=16, fontweight='bold', pad=10)
    ax2.set_xlim(0, 105)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.1f}%', ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=300, bbox_inches='tight')
    print("✓ Prediction visualization saved to 'prediction_result.png'")
    plt.show()
    
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.1%}")
    print(f"{'='*60}")
    print("\nAll Class Probabilities:")
    print("-" * 60)
    for name, prob in zip(class_names, probabilities):
        bar = '█' * int(prob * 50)
        print(f"  {name:12s}: {prob*100:5.2f}% {bar}")
    print(f"{'='*60}\n")
    
    return predicted_class, confidence

if __name__ == '__main__':
    import sys
    
    print("\n" + "="*60)
    print("FACE SHAPE CLASSIFICATION SYSTEM")
    print("Powered by ResNet50 Transfer Learning")
    print("="*60 + "\n")
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'train':
            # Train the model
            main()
        
        elif command == 'test':
            # Test the model on test set
            if not os.path.exists('best_face_shape_model.pth'):
                print("❌ Error: Model file 'best_face_shape_model.pth' not found!")
                print("Please train the model first using: python face_shape.py train")
            else:
                test_model('best_face_shape_model.pth', './face-shape-dataset', device)
        
        elif command == 'predict':
            # Predict on a single image
            if len(sys.argv) < 3:
                print("Usage: python face_shape.py predict <image_path>")
                print("Example: python face_shape.py predict test_image.jpg")
            else:
                image_path = sys.argv[2]
                if not os.path.exists(image_path):
                    print(f"❌ Error: Image file '{image_path}' not found!")
                elif not os.path.exists('best_face_shape_model.pth'):
                    print("❌ Error: Model file 'best_face_shape_model.pth' not found!")
                    print("Please train the model first using: python face_shape.py train")
                else:
                    predict_and_visualize('best_face_shape_model.pth', image_path, device)
        
        else:
            print(f"❌ Unknown command: '{command}'")
            print("\nAvailable commands:")
            print("  train   - Train the model")
            print("  test    - Test the model on test set")
            print("  predict - Predict on a single image")
            print("\nExamples:")
            print("  python face_shape.py train")
            print("  python face_shape.py test")
            print("  python face_shape.py predict my_image.jpg")
    
    else:
        # Default: show usage
        print("Usage: python face_shape.py [command] [arguments]")
        print("\nCommands:")
        print("  train             - Train the model")
        print("  test              - Evaluate model on test set")
        print("  predict <image>   - Predict face shape for an image")
        print("\nExamples:")
        print("  python face_shape.py train")
        print("  python face_shape.py test")
        print("  python face_shape.py predict my_photo.jpg")
        print("\nRunning training by default...")
        print()
        main()