import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, accuracy_score, f1_score,
    roc_curve, auc, balanced_accuracy_score
)

CLASS_NAMES = [
    'pen', 'paper', 'book', 'clock', 'phone', 'laptop',
    'chair', 'desk', 'bottle', 'keychain', 'backpack', 'calculator'
]
n_classes = len(CLASS_NAMES)

# Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class MultiLabelImageFolder(Dataset):
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    def __init__(self, root, class_names, transform=None):
        self.root        = root
        self.class_names = class_names
        self.transform   = transform
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.samples     = self._load_samples()

    def _folder_to_multihot(self, folder_name):
        vec = torch.zeros(len(self.class_names), dtype=torch.float32)
        for token in folder_name.split('_'):
            if token in self.class_to_idx:
                vec[self.class_to_idx[token]] = 1.0
        return vec

    def _load_samples(self):
        samples = []
        for folder in sorted(os.listdir(self.root)):
            folder_path = os.path.join(self.root, folder)
            if not os.path.isdir(folder_path):
                continue
            label = self._folder_to_multihot(folder)
            if label.sum() == 0:
                print(f'  WARNING: folder "{folder}" matched no class names — skipping.')
                continue
            for fname in os.listdir(folder_path):
                if os.path.splitext(fname)[1].lower() in self.VALID_EXTENSIONS:
                    samples.append((os.path.join(folder_path, fname), label))
        print(f'Loaded {len(samples)} images across {len(os.listdir(self.root))} folders.')
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

class TransformSubset(Dataset):
    """Wraps a PyTorch Subset with a specific transform."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        # We need to access the underlying dataset to get the raw image path/label
        # before any transforms are applied
        original_dataset = self.subset.dataset
        idx = self.subset.indices[index]
        img_path, label = original_dataset.samples[idx]
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)


def create_model(num_labels):
    # Upgraded to EfficientNetV2 for SOTA performance.
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    
    # Freeze bottom layers (optional, but standard for transfer learning)
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Unfreeze top layers for fine-tuning
    for param in model.features[-3:].parameters():
        param.requires_grad = True

    in_features = model.classifier[1].in_features
    
    # Rebuild classifier head
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features), # Essential to prevent bias explosion from pos_weight
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_labels)
    )
    return model


def optimize_threshold(y_true, y_probs, class_names):
    """Dynamically finds the optimal F1 threshold for each class"""
    print("Optimizing thresholds on Validation Set...")
    best_thresholds = np.full(len(class_names), 0.5)
    for i, cls in enumerate(class_names):
        # We need a fallback if there are no positives in val set
        if np.sum(y_true[:, i]) == 0:
            continue
            
        precisions, recalls, thresholds = roc_curve(y_true[:, i], y_probs[:, i])
        
        # Suppress warnings for divide by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = (2 * precisions * recalls) / (precisions + recalls)
            f1_scores = np.nan_to_num(f1_scores)
            
        # Prevent the threshold from collapsing to 0.0 or 1.0 (which forces all 1s or all 0s)
        # In highly imbalanced epochs, ROC max F1 can erroneously peak at threshold 0.0
        valid_indices = np.where((thresholds >= 0.1) & (thresholds <= 0.9))[0]
        
        if len(valid_indices) > 0:
            best_idx = valid_indices[np.argmax(f1_scores[valid_indices])]
            best_thresholds[i] = thresholds[best_idx]
        else:
            best_thresholds[i] = 0.5
    print(f"Optimal Thresholds: {best_thresholds}")
    return best_thresholds


def main():
    parser = argparse.ArgumentParser(description="MultiLabel Image Classification Training")
    parser.add_argument("--data_dir", type=str, default="aggregated", help="Path to the dataset directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="TensorBoard log directory")
    parser.add_argument("--model_save_path", type=str, default="best_model.pth", help="Where to save the model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=224, help="Resize dimension for inputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_save_path) or '.', exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = MultiLabelImageFolder(root=args.data_dir, class_names=CLASS_NAMES, transform=None)
    
    total = len(full_dataset)
    if total == 0:
        print(f"Empty dataset at {args.data_dir}. Please try pointing to the right directory.")
        return

    train_size = int(total * 0.70)
    val_size   = int(total * 0.15)
    test_size  = total - train_size - val_size

    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Wrap the subsets with their respective transforms
    train_dataset = TransformSubset(train_subset, transform=train_transform)
    val_dataset   = TransformSubset(val_subset, transform=val_test_transform)
    test_dataset  = TransformSubset(test_subset, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Class weights for Focal / Asymmetric style BCE
    label_counts = torch.zeros(n_classes)
    total_train  = 0
    
    print("Calculating class distribution weights...")
    for _, y in tqdm(train_loader):
        label_counts += y.sum(dim=0)
        total_train  += y.size(0)

    neg_counts = total_train - label_counts
    # Softening the pos_weight using torch.sqrt to prevent precision collapse and boost Exact Match
    class_weights_tensor = torch.sqrt(neg_counts / label_counts.clamp(min=1)).to(device)

    model = create_model(n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)

    writer = SummaryWriter(log_dir=args.log_dir)

    best_val_loss = float('inf')
    patience_counter = 0

    print("\nStarting Training...")
    best_thresholds = np.full(n_classes, 0.5)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for X, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]'):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)

        train_loss = running_loss / len(train_dataset)

        model.eval()
        val_loss_sum = 0.0
        val_probs, val_labels = [], []
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss_sum += loss.item() * X.size(0)
                val_probs.append(torch.sigmoid(outputs).cpu().numpy())
                val_labels.append(y.cpu().numpy())

        val_loss = val_loss_sum / len(val_dataset)
        val_probs = np.vstack(val_probs)
        val_labels = np.vstack(val_labels)
        
        # Calculate naive 0.5 accuracy for fast tracking
        val_acc = (val_probs > 0.5).astype(float)
        acc_score = accuracy_score(val_labels, val_acc)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val_naive', acc_score, epoch)

        print(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_naive_acc={acc_score:.4f}')

        # Fixed Early Stopping Checkpoint Logic
        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.model_save_path)
            print(f"  * Checkpoint Saved: best_model.pth (Val Loss Improved)")
            
            # Recalculate dynamic thresholds for best working model
            best_thresholds = optimize_threshold(val_labels, val_probs, CLASS_NAMES)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping triggered! No improvement for {args.patience} epochs.')
                break
                
    writer.close()

    print("\n--- Evaluating Best Model ---")
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(device)
    model.eval()

    test_loss = 0.0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            test_loss += loss.item() * X.size(0)
            all_probs.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(y.cpu().numpy())

    test_loss /= len(test_dataset)
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    # Use dynamically calculated thresholds
    all_predicts = (all_probs > best_thresholds).astype(float)
    
    print(f'Test Loss: {test_loss:.4f}')
    print('\n--- Per-class report ---')
    print(classification_report(all_labels, all_predicts, target_names=CLASS_NAMES, digits=4, zero_division=0))

    print('\n--- Aggregated metrics ---')
    for avg in ('macro', 'micro', 'samples', 'weighted'):
        p = precision_score(all_labels, all_predicts, average=avg, zero_division=0)
        r = recall_score(all_labels,    all_predicts, average=avg, zero_division=0)
        f = f1_score(all_labels,        all_predicts, average=avg, zero_division=0)
        print(f'  [{avg:9s}]  Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}')

    subset_acc = accuracy_score(all_labels, all_predicts)
    print(f'\nSubset (exact-match) Accuracy: {subset_acc:.4f}')
    hamming_acc = (all_labels == all_predicts).mean()
    print(f'Hamming Accuracy (element-wise): {hamming_acc:.4f}')

if __name__ == '__main__':
    main()
