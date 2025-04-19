import warnings
warnings.filterwarnings("ignore", message="You should probably TRAIN this model")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Force CPU usage
device = torch.device("cpu")

# Load dataset
try:
    dataset = load_dataset("json", data_files="synthetic_dataset_full.json")["train"]
    print("Dataset loaded successfully!")
    print(f"Dataset size: {len(dataset)} examples")
    print("First example:", dataset[0])
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Split dataset
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_data = split_dataset["train"]
val_data = split_dataset["test"]
print(f"Training set: {len(train_data)} examples")
print(f"Validation set: {len(val_data)} examples")

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# SFT Training
baseline_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
baseline_model.to(device)
baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.0001)

def train_sft(model, dataset, epochs=10, batch_size=8):
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        indices = np.random.permutation(len(dataset))
        progress_bar = tqdm(range(0, len(dataset), batch_size), desc=f"SFT Epoch {epoch+1}")
        
        for i in progress_bar:
            batch_indices = indices[i:i+batch_size]
            batch = dataset[batch_indices]
            
            inputs = tokenizer(
                [f"{p} [SEP] {r1} [SEP] {r2}" for p, r1, r2 in zip(batch["prompt"], batch["response1"], batch["response2"])],
                return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(device)
            
            labels = torch.tensor(batch["preferred"], dtype=torch.long).to(device)
            
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            
            baseline_optimizer.zero_grad()
            loss.backward()
            baseline_optimizer.step()
            
            current_loss = loss.item()
            epoch_loss += current_loss
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})
            
        avg_epoch_loss = epoch_loss / (len(dataset) // batch_size)
        losses.append(avg_epoch_loss)
        print(f"SFT Epoch {epoch+1}, Average Loss: {avg_epoch_loss:.4f}")
    
    return losses

# Modified SPAC Training with learning rate as parameter
def train_spac(model, dataset, lambda_=1.0, epochs=10, batch_size=8, learning_rate=0.01):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    pref_losses = []
    reg_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_pref_loss = 0
        epoch_reg_loss = 0
        indices = np.random.permutation(len(dataset))
        progress_bar = tqdm(range(0, len(dataset), batch_size), desc=f"SPAC Epoch {epoch+1}")
        
        for i in progress_bar:
            batch_indices = indices[i:i+batch_size]
            batch = dataset[batch_indices]
            
            inputs1 = tokenizer(
                [f"{p} [SEP] {r1}" for p, r1 in zip(batch["prompt"], batch["response1"])],
                return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(device)
            
            inputs2 = tokenizer(
                [f"{p} [SEP] {r2}" for p, r2 in zip(batch["prompt"], batch["response2"])],
                return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(device)
            
            score1 = model(**inputs1).logits[:, 1]
            score2 = model(**inputs2).logits[:, 1]
            
            score_diff = score1 - score2
            
            preferred = torch.tensor(batch["preferred"], dtype=torch.float).to(device)
            
            # SPAC loss components:
            preference_loss = -torch.mean(preferred * F.logsigmoid(score_diff) + 
                                         (1 - preferred) * F.logsigmoid(-score_diff))
            
            regularization = lambda_ * torch.mean(F.relu(-score_diff))
            
            loss = preference_loss + regularization
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            current_pref_loss = preference_loss.item()
            current_reg_loss = regularization.item()
            
            epoch_loss += current_loss
            epoch_pref_loss += current_pref_loss
            epoch_reg_loss += current_reg_loss
            
            progress_bar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "pref_loss": f"{current_pref_loss:.4f}", 
                "reg_loss": f"{current_reg_loss:.4f}"
            })
        
        batch_count = len(dataset) // batch_size
        avg_epoch_loss = epoch_loss / batch_count
        avg_pref_loss = epoch_pref_loss / batch_count
        avg_reg_loss = epoch_reg_loss / batch_count
        
        losses.append(avg_epoch_loss)
        pref_losses.append(avg_pref_loss)
        reg_losses.append(avg_reg_loss)
        
        print(f"SPAC Epoch {epoch+1}, Total Loss: {avg_epoch_loss:.4f}, "
              f"Preference Loss: {avg_pref_loss:.4f}, Regularization: {avg_reg_loss:.4f}")
    
    return losses, pref_losses, reg_losses

# Evaluation function 
def evaluate(model, data, batch_size=8):
    model.eval()
    correct = 0
    total = 0
    all_scores = []
    
    with torch.no_grad():
        progress_bar = tqdm(range(0, len(data), batch_size), desc="Evaluating")
        
        for i in progress_bar:
            batch = data[i:i+batch_size]
            
            inputs1 = tokenizer(
                [f"{p} [SEP] {r1}" for p, r1 in zip(batch["prompt"], batch["response1"])],
                return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(device)
            
            inputs2 = tokenizer(
                [f"{p} [SEP] {r2}" for p, r2 in zip(batch["prompt"], batch["response2"])],
                return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(device)
            
            score1 = model(**inputs1).logits[:, 1]
            score2 = model(**inputs2).logits[:, 1]
            
            # Predict preference based on scores
            predictions = (score1 > score2).int()
            gold_labels = torch.tensor(batch["preferred"], device=device)
            
            for j in range(len(predictions)):
                all_scores.append({
                    "pred": predictions[j].item(),
                    "gold": gold_labels[j].item(),
                    "score1": score1[j].item(),
                    "score2": score2[j].item(),
                    "diff": (score1[j] - score2[j]).item()
                })
            
            correct += (predictions == gold_labels).sum().item()
            total += len(gold_labels)
            
            progress_bar.set_postfix({"acc": f"{correct/total:.4f}"})
    
    return correct / total, all_scores

# Using fine-grained lambda values and learning rate tuning
def run_experiments(train_epochs=10):
    results = {
        "sft": {"accuracy": None, "losses": None},
        "spac": {}
    }
    
    print("\n=== Training SFT Model ===")
    sft_losses = train_sft(baseline_model, train_data, epochs=train_epochs)
    sft_accuracy, sft_scores = evaluate(baseline_model, val_data)
    results["sft"]["accuracy"] = sft_accuracy
    results["sft"]["losses"] = sft_losses
    
    lambda_values = [0.5, 0.8, 1.0, 1.2, 1.5]  # Fine-grained lambda testing around 1.0
    learning_rates = [0.001, 0.01]  # Lower learning rates than the paper's 0.1
    
    # Perform grid search over lambda and learning rate values
    for lr in learning_rates:
        for lambda_ in lambda_values:
            config_name = f"λ={lambda_}_lr={lr}"
            print(f"\n=== Training SPAC Model with {config_name} ===")
            
            spac_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
            spac_model.to(device)
            
            spac_losses, pref_losses, reg_losses = train_spac(
                spac_model, train_data, lambda_=lambda_, 
                epochs=train_epochs, learning_rate=lr
            )
            
            spac_accuracy, spac_scores = evaluate(spac_model, val_data)
            
            results["spac"][config_name] = {
                "accuracy": spac_accuracy,
                "losses": spac_losses,
                "pref_losses": pref_losses,
                "reg_losses": reg_losses,
                "lambda": lambda_,
                "lr": lr
            }
    
    return results

def visualize_results(results):
    # 1. Accuracy comparison
    plt.figure(figsize=(12, 6))
    
    plt.axhline(y=0.5, color="red", linestyle="--", label="Random Baseline (0.5)")
    
    bar_positions = [0]  
    bar_heights = [results["sft"]["accuracy"]]
    bar_labels = ["SFT"]
    bar_colors = ["blue"]
    
    spac_configs = list(results["spac"].keys())
    spac_accuracies = [results["spac"][config]["accuracy"] for config in spac_configs]
    
    for i, config in enumerate(spac_configs):
        bar_positions.append(i + 1)
        bar_heights.append(results["spac"][config]["accuracy"])
        bar_labels.append(config)
        bar_colors.append("green")
    
    plt.bar(bar_positions, bar_heights, color=bar_colors, width=0.6)
    plt.xticks(bar_positions, bar_labels, rotation=45, ha="right")
    
    plt.ylabel("Validation Accuracy")
    plt.title("Preference Learning: SFT vs SPAC Accuracy Comparison")
    plt.ylim(0.45, max(bar_heights) + 0.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("accuracy_comparison_grid.png")
    plt.show()
    
    # 2. Loss comparison by learning rate
    plt.figure(figsize=(15, 10))
    
    lr_configs = {}
    for config, data in results["spac"].items():
        lr = data["lr"]
        if lr not in lr_configs:
            lr_configs[lr] = []
        lr_configs[lr].append(config)
    
    for i, (lr, configs) in enumerate(lr_configs.items()):
        plt.subplot(len(lr_configs), 1, i+1)
        
        plt.plot(results["sft"]["losses"], 'k--', label="SFT Loss")
        
        for config in configs:
            lambda_ = results["spac"][config]["lambda"]
            plt.plot(results["spac"][config]["losses"], 
                     label=f"SPAC λ={lambda_}, lr={lr}")
        
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss Comparison (lr={lr})")
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig("loss_comparison_by_lr.png")
    plt.show()
    
    # 3. Components of best model
    best_config = max(results["spac"].items(), key=lambda x: x[1]["accuracy"])[0]
    best_data = results["spac"][best_config]
    
    plt.figure(figsize=(10, 6))
    plt.plot(best_data["pref_losses"], 'g-', label="Preference Loss")
    plt.plot(best_data["reg_losses"], 'r-', label="Regularization Loss")
    plt.plot(best_data["losses"], 'b-', label="Total Loss")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"SPAC Loss Components ({best_config})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("best_spac_loss_components.png")
    plt.show()
    
    # 4. Heatmap of lambda vs learning rate performance
    plt.figure(figsize=(10, 6))
    
    lambda_values = sorted(list(set([results["spac"][config]["lambda"] for config in results["spac"]])))
    lr_values = sorted(list(set([results["spac"][config]["lr"] for config in results["spac"]])))
    
    accuracy_matrix = np.zeros((len(lr_values), len(lambda_values)))
    
    for i, lr in enumerate(lr_values):
        for j, lambda_ in enumerate(lambda_values):
            config_name = f"λ={lambda_}_lr={lr}"
            if config_name in results["spac"]:
                accuracy_matrix[i, j] = results["spac"][config_name]["accuracy"]
    
    plt.imshow(accuracy_matrix, cmap='viridis')
    plt.colorbar(label='Validation Accuracy')
    
    plt.xticks(np.arange(len(lambda_values)), [f"λ={l}" for l in lambda_values])
    plt.yticks(np.arange(len(lr_values)), [f"lr={l}" for l in lr_values])
    
    for i in range(len(lr_values)):
        for j in range(len(lambda_values)):
            plt.text(j, i, f"{accuracy_matrix[i, j]:.3f}", 
                     ha="center", va="center", color="white" if accuracy_matrix[i, j] < 0.6 else "black")
    
    plt.xlabel("Lambda (Regularization Strength)")
    plt.ylabel("Learning Rate")
    plt.title("SPAC Performance Heatmap")
    plt.tight_layout()
    plt.savefig("spac_parameter_heatmap.png")
    plt.show()

if __name__ == "__main__":
    TRAIN_EPOCHS = 10  
    
    print("=== Running SFT vs SPAC Comparison with Fine-Grained Parameter Tuning ===")
    results = run_experiments(train_epochs=TRAIN_EPOCHS)
    
    print("\n=== Final Results ===")
    print(f"SFT Accuracy: {results['sft']['accuracy']:.4f}")
    
    for config, data in results["spac"].items():
        print(f"SPAC ({config}) Accuracy: {data['accuracy']:.4f}")
    
    # Find best configuration
    best_config = max(results["spac"].items(), key=lambda x: x[1]["accuracy"])[0]
    best_spac_acc = results["spac"][best_config]["accuracy"]
    
    improvement = (best_spac_acc - results["sft"]["accuracy"]) * 100
    
    print(f"\n=== Detailed Analysis ===")
    print(f"Best SPAC configuration: {best_config}")
    print(f"SPAC outperforms SFT by {improvement:.2f}% absolute accuracy")
    
    visualize_results(results)