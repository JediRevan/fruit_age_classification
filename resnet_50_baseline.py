import os
import numpy as np
import pandas as pd
from spectral.io.envi import open as envi_open
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --- Constants ---
DATA_DIR = "./VIS"  # Update with your dataset path
IMG_SIZE = 224  # Resize images to ResNet-compatible size
BATCH_SIZE = 16
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Helper Functions ---
def read_hyperspectral_image(hdr_file):
    """Load and preprocess hyperspectral image from .hdr and its associated .bin file."""
    bin_file = hdr_file.replace(".hdr", ".bin")  # Ensure .bin file path
    try:
        img = envi_open(hdr_file, image=bin_file).load()
        img = np.mean(img, axis=2)
        img = np.stack([img] * 3, axis=-1)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = preprocess(img)
        return img
    except Exception as e:
        raise RuntimeError(f"Error reading image: {hdr_file}") from e

def unnormalize(img):
    """Reverse the normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean  # Reapply standardization reversal
    return img.clamp(0, 1)  # Ensure values are in [0, 1]

def plot_image_pair(img1, img2, title1="Image 1", title2="Image 2"):
    """Plot two images side by side for visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, img, title in zip(axes, [img1, img2], [title1, title2]):
        img = unnormalize(img)  # Unnormalize before display
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(title)
        ax.axis("off")
    plt.show()

def visualize_sample_predictions(model, test_loader, num_samples=5):
    """Visualize sample predictions with true and predicted labels."""
    model.eval()
    samples_shown = 0
    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.to(DEVICE)
            predictions = model(img1, img2).squeeze().cpu().numpy()
            labels = labels.cpu().numpy()

            for i in range(len(labels)):
                if samples_shown >= num_samples:
                    return
                print(f"True Label: {labels[i]:.2f}, Predicted: {predictions[i]:.2f}")
                plot_image_pair(img1[i].cpu(), img2[i].cpu())
                samples_shown += 1


def analyze_errors(model, test_loader, threshold=2):
    """Analyze errors where predictions deviate significantly from true labels."""
    model.eval()
    significant_errors = []
    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.to(DEVICE)
            predictions = model(img1, img2).squeeze()
            errors = torch.abs(predictions - labels)
            significant_errors.extend([(img1[i].cpu(), img2[i].cpu(), labels[i].item(), predictions[i].item())
                                       for i in range(len(errors)) if errors[i] > threshold])
    print(f"Significant Errors: {len(significant_errors)}")
    return significant_errors


def visualize_errors(errors, num_samples=5):
    """Visualize significant errors with true and predicted labels."""
    for i, (img1, img2, true_label, pred_label) in enumerate(errors[:num_samples]):
        print(f"True Label: {true_label:.2f}, Predicted: {pred_label:.2f}")
        plot_image_pair(img1, img2, title1="Image 1 (Error)", title2="Image 2 (Error)")


def extract_embeddings(model, dataset):
    """Extract intermediate embeddings from the model for visualization."""
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for img1, img2, label in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False):
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
            emb1 = model.extract_features(img1).cpu().numpy()  # Extract intermediate features
            emb2 = model.extract_features(img2).cpu().numpy()
            embeddings.extend(np.abs(emb1 - emb2))  # Compute absolute difference as embedding
            labels.extend(label.cpu().numpy())
    return np.array(embeddings), np.array(labels)


def visualize_embeddings(embeddings, labels, method="PCA"):
    """Visualize embeddings using PCA or t-SNE."""
    if embeddings.shape[0] < 2 or embeddings.shape[1] < 2:
        print("Insufficient data for PCA or t-SNE. Skipping embedding visualization.")
        return

    if method == "PCA":
        n_components = min(2, embeddings.shape[1], embeddings.shape[0])
        reducer = PCA(n_components=n_components)
    elif method == "t-SNE":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Method should be 'PCA' or 't-SNE'.")

    reduced_embeddings = reducer.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="viridis", s=15)
    plt.colorbar(scatter, label="Day Difference")
    plt.title(f"Embedding Visualization ({method})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

def get_image_metadata(file_name):
    """
    Parse the file name to extract mango ID and day.
    Assumes a file naming convention like 'mango_day_10_m3_01_front.bin'.
    """
    try:
        parts = file_name.split("_")
        day = int(parts[2])  # Extract day (e.g., 'day_10' -> 10)
        mango_id = parts[4].split(".")[0]  # Extract mango ID (e.g., '01_front' -> '01')
        return mango_id, day
    except (IndexError, ValueError) as e:
        print(f"Failed to parse metadata from: {file_name}")
        raise e

def create_pairs(data):
    """Generate pairs of images from the same mango but different days."""
    pairs = []
    labels = []
    for mango_id, group in data.groupby("mango_id"):
        group = group.sort_values("day")
        files = group["file"].tolist()
        days = group["day"].tolist()
        for i in range(len(files) - 1):
            for j in range(i + 1, len(files)):
                if days[i] != days[j]:
                    pairs.append((files[i], files[j]))
                    labels.append(abs(days[j] - days[i]))
    return pairs, labels

def validate_pairs_and_labels(pairs, labels):
    """Validate pairs and corresponding labels."""
    valid_pairs = []
    valid_labels = []
    for i, (file1, file2) in enumerate(pairs):
        hdr1, hdr2 = file1.replace(".bin", ".hdr"), file2.replace(".bin", ".hdr")
        bin1, bin2 = hdr1.replace(".hdr", ".bin"), hdr2.replace(".hdr", ".bin")
        if os.path.exists(hdr1) and os.path.exists(hdr2) and os.path.exists(bin1) and os.path.exists(bin2):
            valid_pairs.append((file1, file2))
            valid_labels.append(labels[i])
    return valid_pairs, valid_labels

def print_sample_pairs(pairs, labels, dataset_name, num_samples=5):
    print(f"\nSample pairs and labels from {dataset_name} set:")
    for i in range(min(num_samples, len(pairs))):
        file1, file2 = pairs[i]
        label = labels[i]
        print(f"Pair {i+1}: File 1: {file1}, File 2: {file2}, Label: {label}")
    print("\n")

# --- Data Preparation ---
data = []
for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".bin"):
            try:
                mango_id, day = get_image_metadata(file)
                hdr_file = os.path.join(root, file.replace(".bin", ".hdr"))
                if os.path.exists(hdr_file):
                    data.append({"file": os.path.join(root, file), "hdr": hdr_file, "mango_id": mango_id, "day": day})
            except ValueError:
                print(f"Skipping file due to error: {file}")

data = pd.DataFrame(data)

train_ids, test_ids = train_test_split(data["mango_id"].unique(), test_size=0.2, random_state=42)
val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)

train_data = data[data["mango_id"].isin(train_ids)]
val_data = data[data["mango_id"].isin(val_ids)]
test_data = data[data["mango_id"].isin(test_ids)]

train_pairs, train_labels = create_pairs(train_data)
val_pairs, val_labels = create_pairs(val_data)
test_pairs, test_labels = create_pairs(test_data)

train_pairs, train_labels = validate_pairs_and_labels(train_pairs, train_labels)
val_pairs, val_labels = validate_pairs_and_labels(val_pairs, val_labels)
test_pairs, test_labels = validate_pairs_and_labels(test_pairs, test_labels)

print_sample_pairs(train_pairs, train_labels, "Train", num_samples=5)
print_sample_pairs(val_pairs, val_labels, "Validation", num_samples=5)
print_sample_pairs(test_pairs, test_labels, "Test", num_samples=5)

# --- PyTorch Dataset ---
class MangoPairDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        file1, file2 = self.pairs[idx]
        hdr1, hdr2 = file1.replace(".bin", ".hdr"), file2.replace(".bin", ".hdr")
        img1 = read_hyperspectral_image(hdr1)
        img2 = read_hyperspectral_image(hdr2)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img1, img2, label

train_loader = DataLoader(MangoPairDataset(train_pairs, train_labels), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(MangoPairDataset(val_pairs, val_labels), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(MangoPairDataset(test_pairs, test_labels), batch_size=BATCH_SIZE, shuffle=False)

# --- ResNet-50 Model ---
class ResNet50Regression(nn.Module):
    def __init__(self):
        super(ResNet50Regression, self).__init__()
        self.base_model = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V2")
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1)  # Regression layer

    def forward(self, img1, img2):
        x1 = self.base_model(img1)
        x2 = self.base_model(img2)
        diff = torch.abs(x1 - x2)
        return diff

    def extract_features(self, x):
        """Extract features before the FC layer."""
        # Remove the final FC layer and get the features
        features = nn.Sequential(*list(self.base_model.children())[:-1])
        return features(x).view(x.size(0), -1)  # Flatten the output

model = ResNet50Regression().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- Training Loop ---
def train_model(model, train_loader, val_loader, epochs, patience=10):
    best_val_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for img1, img2, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(img1, img2).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.to(DEVICE)
                outputs = model(img1, img2).squeeze()
                val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping!")
                break

# Train the model
train_model(model, train_loader, val_loader, EPOCHS, patience=10)

# --- Evaluation ---
def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.to(DEVICE)
            outputs = model(img1, img2).squeeze()
            test_loss += criterion(outputs, labels).item()
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

evaluate_model(model, test_loader)

# Example: Qualitative Analysis after Evaluation
print("Visualizing Sample Predictions:")
visualize_sample_predictions(model, test_loader, num_samples=5)

errors = analyze_errors(model, test_loader, threshold=2)
visualize_errors(errors, num_samples=5)

# Extract embeddings using the updated model
print("Extracting and Visualizing Embeddings:")
test_dataset = MangoPairDataset(test_pairs, test_labels)
embeddings, labels = extract_embeddings(model, test_dataset)

# Perform dimensionality reduction
if embeddings.shape[0] > 1 and embeddings.shape[1] > 1:
    visualize_embeddings(embeddings, labels, method="PCA")
    visualize_embeddings(embeddings, labels, method="t-SNE")
else:
    print("Insufficient data for PCA or t-SNE. Skipping embedding visualization.")

