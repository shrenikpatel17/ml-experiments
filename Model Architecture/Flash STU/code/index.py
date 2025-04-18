import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

# data generation
def generate_sequence(length=1000, noise_std=0.05):
    t = np.arange(length)
    x = np.zeros(length)
    for i in range(1, length):
        x[i] = 0.99 * x[i-1] + np.sin(0.1 * t[i]) + np.random.normal(0, noise_std)
    return x

def create_dataset(num_sequences=500, seq_length=1000, window=50):
    sequences = [generate_sequence(seq_length) for _ in range(num_sequences)]
    X, Y = [], []
    for seq in sequences:
        for i in range(len(seq) - window):
            X.append(seq[i:i+window])
            Y.append(seq[i+window])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

# compute hankel filters
def compute_hankel_filters(L=50, num_alpha=100, num_filters=10):
    alphas = np.linspace(0, 1, num_alpha)
    mu_matrix = np.array([[alpha ** i for i in range(L)] for alpha in alphas])
    Z = (mu_matrix.T @ mu_matrix) / num_alpha
    eigenvalues, eigenvectors = np.linalg.eigh(Z)
    idx = np.argsort(eigenvalues)[::-1]
    filters = eigenvectors[:, idx[:num_filters]]
    return filters.T 

class STULayer(nn.Module):
    def __init__(self, fixed_filters, out_channels=1):
        super().__init__()
        num_filters, L = fixed_filters.shape
        self.register_buffer('filters', torch.tensor(fixed_filters, dtype=torch.float32).view(num_filters, 1, L))
        self.proj = nn.Linear(num_filters, out_channels)

    def forward(self, x):
        conv_out = F.conv1d(x, self.filters, padding=self.filters.shape[2]//2)
        conv_out = conv_out.transpose(1, 2)
        out = self.proj(conv_out)
        return out

class SimpleAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        weights = F.softmax(scores, dim=-1)
        out = torch.bmm(weights, V)
        return out

# predictors
class STUPredictor(nn.Module):
    def __init__(self, stu_layer):
        super().__init__()
        self.stu = stu_layer

    def forward(self, x):
        out = self.stu(x)
        return out[:, -1, :]

class AttentionPredictor(nn.Module):
    def __init__(self, d_model, window):
        super().__init__()
        self.lift = nn.Linear(1, d_model)
        self.attn = SimpleAttention(d_model)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.lift(x)
        attn_out = self.attn(x)
        return self.proj(attn_out[:, -1, :])

def train_model(model, optimizer, dataloader, criterion, epochs=20):
    model.train()
    losses = []
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        avg = total / len(dataloader.dataset)
        losses.append(avg)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg:.6f}")
    return losses

if __name__ == "__main__":
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    num_sequences = 500
    seq_len = 300
    window = 50
    num_alpha = 100
    num_filters = 10
    batch_size = 64
    lr = 1e-3
    epochs = 20

    X, Y = create_dataset(num_sequences, seq_len, window)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    Y_t = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
    ds = TensorDataset(X_t, Y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    filters = compute_hankel_filters(window, num_alpha, num_filters)

    stu_layer = STULayer(filters, out_channels=1)
    stu_model = STUPredictor(stu_layer)
    attn_model = AttentionPredictor(d_model=num_filters, window=window)

    opt_stu = optim.Adam(stu_model.parameters(), lr=lr)
    opt_attn = optim.Adam(attn_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("Training STU model...")
    losses_stu = train_model(stu_model, opt_stu, loader, criterion, epochs)
    print("Training Attention model...")
    losses_attn = train_model(attn_model, opt_attn, loader, criterion, epochs)

    plt.figure(figsize=(10, 6))
    plt.plot(losses_stu, label="STU")
    plt.plot(losses_attn, label="Attention")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Comparison: STU vs Attention")
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(0, 20, dtype=int))
    
    plot_filename = os.path.join(results_dir, f"loss_comparison_{timestamp}.png")
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.show()

    # sample predictions
    stu_model.eval()
    attn_model.eval()
    xb, yb = next(iter(loader))
    with torch.no_grad():
        p_stu = stu_model(xb)
        p_attn = attn_model(xb)
    
    # convert predictions to numpy for JSON serialization
    true_values = yb[:5].squeeze().numpy().tolist()
    stu_predictions = p_stu[:5].squeeze().numpy().tolist()
    attn_predictions = p_attn[:5].squeeze().numpy().tolist()
    
    print("True:", true_values)
    print("STU:", stu_predictions)
    print("Attn:", attn_predictions)
    
    results = {
        "hyperparameters": {
            "num_sequences": num_sequences,
            "seq_len": seq_len,
            "window": window,
            "num_alpha": num_alpha,
            "num_filters": num_filters,
            "batch_size": batch_size,
            "learning_rate": lr,
            "epochs": epochs
        },
        "training_losses": {
            "stu": losses_stu,
            "attention": losses_attn
        },
        "sample_predictions": {
            "true_values": true_values,
            "stu_predictions": stu_predictions,
            "attention_predictions": attn_predictions
        },
        "dataset_info": {
            "X_shape": list(X.shape),
            "Y_shape": list(Y.shape)
        }
    }
    
    json_filename = os.path.join(results_dir, f"experiment_results_{timestamp}.json")
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {json_filename}")
