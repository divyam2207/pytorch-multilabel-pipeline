#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" CAI6108 -- Project -- eval.py

This file contains the evaluation code to load a trained model from a specified checkpoint, evaluate it on a test dataset, and report performance. We will run this script on the test dataset to evaluate the model's performance.
use the command:

python eval.py --model_path YOUR_SAVED_MODEL --test_data project_test_data --group_id YOUR_GROUP_ID --project_title "YOUR_PROJECT_TITLE"

Note that you need to write the model creation function and call it in the load_trained_model function. You may also
need to change the predict function (e.g., if your pipeline is not compatible with the provided implementation).
Please test the model creation function and the model loading function and predict function to make sure they work. If we cannot load or evaluate your model, we will apply a penalty.

"""

import sys
import time
import argparse
import re

from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.utils.data import DataLoader, Dataset

LABEL_ORDER = ["pen", "paper", "book", "clock", "phone", "laptop", "chair", "desk", "bottle", "keychain", "backpack", "calculator"]
VALID_LABELS = set(LABEL_ORDER)
LABEL_TO_IDX = {label: i for i, label in enumerate(LABEL_ORDER)}
IMG_RE = re.compile(r"^img(\S+)\.png$", re.IGNORECASE)

######### Classes ##########

class CustomDirectoryLayoutDataset(Dataset):
    def __init__(self, root, transform=None, separator='_', classes=LABEL_ORDER):
        self.root = Path(root)
        self.transform = transform
        self.separator = separator
        self.classes = classes
        self.num_classes = len(self.classes)
        self.samples = []

        for subdir in self.root.iterdir():
            if not subdir.is_dir():
                continue

            labels = subdir.name.split(self.separator)

            # ignore any files/directory that does not fit the expected pattern
            if not labels or any(label not in VALID_LABELS for label in labels):
                continue
            if len(labels) != len(set(labels)):
                continue

            # multilabel so the target will have potentially multiple entries set to 1
            target = torch.zeros(self.num_classes, dtype=torch.float32)
            for i, label in enumerate(LABEL_ORDER):
                if label in labels:
                    target[i] = 1.0

            assert target.sum() > 0

            for path in subdir.iterdir():
                if path.is_file() and IMG_RE.match(path.name):
                    self.samples.append((path, target.clone()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


######### Model ##########

def create_model(num_labels):
    # Pass weights=None so PyTorch doesn't try to download weights from the internet.
    # The actual weights will be loaded from the saved checkpoint in load_trained_model.
    model = efficientnet_v2_s(weights=None)

    for param in model.features.parameters():
        param.requires_grad = False

    for param in model.features[-3:].parameters():
        param.requires_grad = True

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_labels)
    )

    return model


######### Functions #########

def load_test_dataset(data_dir, batch_size, num_workers, image_size, shuffle=False):
    """
    Loads the test dataset from a given directory. The directory must contain subfolders
    for each class (like in training). Applies only evaluation transforms.

    Args:
        data_dir (str): Path to test data folder.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        image_size (int): Desired image size.

    Returns:
        DataLoader: DataLoader object for the test dataset.
    """
    test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    test_dataset = CustomDirectoryLayoutDataset(root=data_dir, transform=test_transforms)
    assert len(test_dataset) > 0, f"Empty dataset found ({data_dir})."

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=True)
    return test_loader


def load_trained_model(model_path, num_labels, device, image_size):
    """
    Builds your model architecture, adjusts the classification head to
    the given number of classes, and loads the trained model weights from a local file.

    Args:
        model_path (str): Path to the trained model checkpoint.
        num_labels (int): Number of output labels. (Should be 12 but left for consistency.)
        device (str): Device for model loading ('cuda' or 'cpu').
        image_size (int): desired input image size. (Not used here but kept for consistency.)

    Returns:
        model: The model loaded on device.
    """

    model = create_model(num_labels=num_labels)  

    # Load local state dictionary
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Move the model to the specified device and set evaluation mode.
    model = model.to(device)
    model.eval()

    return model


def predict(model, x, threshold=0.5):
    """
    Computes the predicted labels for a batch of input images.

    Args:
        model: Your trained model
        x (torch.Tensor): Input batch of images.
        threshold: used to get labels from probs

    Returns:
        torch.Tensor: Predicted labels (for the given threshold).
        torch.Tensor: Predicted probabilities.
        torch.Tensor: Logits.
    """

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
    return preds, probs, logits     # the code needs to return all of this for evaluate_model() to work!


def evaluate_model(model, test_loader, device, threshold=0.5):
    """
    Evaluates the model on the test dataset.

    Args:
        model: Your trained model
        test_loader (DataLoader): DataLoader for the test dataset.
        device (str): Device used for evaluation.

    Returns:
        tuple: (test_loss, test_accuracy)
    """
    criterion = nn.BCEWithLogitsLoss()  # again this assumes the model output is logits
    running_loss = 0.0
    total_samples = 0  

    all_probs = []; all_preds = []; all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # call predict()
            preds, probs, logits = predict(model, images, threshold=threshold)

            # compute loss
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # save stuff for metrics
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    test_loss = running_loss / total_samples

    all_probs = torch.cat(all_probs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    exact_match = (all_preds == all_labels).all(dim=1).float().mean().item() # exact matching on every label
    hamming_acc = (all_preds == all_labels).float().mean().item()   # each label treated independently -> average acc

    # this is for IoU also called Jaccard Index
    intersection = (all_preds * all_labels).sum(dim=1)
    union = ((all_preds + all_labels) > 0).float().sum(dim=1)
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    mean_iou = iou.mean().item()

    # sklearn style per-class metrics
    tp = ((all_preds == 1) & (all_labels == 1)).sum().float()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().float()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().float()

    precision_micro = (tp / (tp + fp + 1e-8)).item()
    recall_micro = (tp / (tp + fn + 1e-8)).item()
    f1_micro = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()

    metrics = {"loss": test_loss, "exact_match": exact_match, "hamming_acc": hamming_acc,
        "mean_iou": mean_iou, "precision_micro": precision_micro, "recall_micro": recall_micro,"f1_micro": f1_micro}
    return metrics


######### Main() #########

if __name__ == "__main__":
    exit_code = 0  # reassign a value for errors

    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Eval script for CAI6108 project")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint (e.g., models/trained_model.pth)")
    parser.add_argument("--test_data", type=str, default="project_test_data",
                        help="Directory containing the test dataset with class subfolders")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for DataLoader")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for eval")
    parser.add_argument("--group_id", type=int, required=True, help="Project Group ID (non-negative integer)")
    parser.add_argument("--project_title", type=str, required=True, help="Project Title (at least 4 characters)")

    args = parser.parse_args()

    project_group_id = args.group_id
    project_title = args.project_title

    # Validate required parameters.
    assert project_group_id >= 0, "Group ID must be non-negative"
    assert len(project_title) >= 4, "Project title must be at least 4 characters long"

    # Keep track of time.
    st = time.time()

    # Header.
    print('\n---------- [Eval] (Project: {}, Group: {}) ---------'.format(project_title, project_group_id))

    # Determine the device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Evaluation device:", device)

    # Load test data.
    test_loader = load_test_dataset(args.test_data, args.batch_size, args.num_workers, args.image_size)

    # Grab number of classes from the test dataset. (Should be 12)
    num_classes = len(test_loader.dataset.classes)
    print("Number of classes:", num_classes)

    # Load the trained model from the given checkpoint.
    model = load_trained_model(args.model_path, num_classes, device, args.image_size)
    print("Model loaded successfully from:", args.model_path)

    # Evaluate the model on test data.
    test_metrics = evaluate_model(model, test_loader, device, threshold=args.threshold)
    print(f"Metrics: {test_metrics}")

    # Elapsed time.
    et = time.time()
    elapsed = et - st
    print('---------- [Eval] Elapsed time -- total: {:.1f} seconds ---------\n'.format(elapsed))

    sys.exit(exit_code)