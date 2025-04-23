import torch
import numpy as np
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LAYER_IDX = 16  
SAE_DIM = 1024  
TOP_K = 50  
SAVE_DIR = "sae_analysis"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32  #

prompt_pairs = [
    [
        "Write me a Python function that reverses a string.",
        "Can you show me Python code to reverse a string?"
    ],
    [
        "How do I sort a list in Python?",
        "Write a Python function to sort a list of numbers."
    ],
    [
        "Explain how to implement binary search in Python.",
        "Show me a Python implementation of binary search algorithm."
    ],
    [
        "How can I read a file in Python?",
        "Write code to open and read a file in Python."
    ],
    [
        "Create a Python class representing a simple calculator.",
        "Implement a calculator class in Python with basic arithmetic operations."
    ],
    [
        "How do I remove duplicates from a list in Python?",
        "Write a function to eliminate duplicate elements from a Python list."
    ],
    [
        "Write a Python script to count word frequency in a text.",
        "Show me how to count the occurrence of each word in a text using Python."
    ],
    [
        "Implement a function to check if a string is a palindrome.",
        "Write Python code to determine whether a string reads the same backward as forward."
    ],
    [
        "How do I create a web server in Python?",
        "Write a simple HTTP server using Python."
    ],
    [
        "Explain how to use list comprehensions in Python.",
        "Show examples of Python list comprehensions for various operations."
    ],
    [
        "How can I connect to a SQL database from Python?",
        "Write Python code to establish a connection with a SQL database."
    ],
    [
        "Write a Python function to find the factorial of a number.",
        "Implement a factorial calculator in Python."
    ],
    [
        "How do I create a RESTful API with Flask?",
        "Write code for a simple Flask REST API."
    ],
    [
        "Implement a Python decorator for timing function execution.",
        "Create a decorator in Python that measures how long a function takes to run."
    ],
    [
        "How do I parse JSON data in Python?",
        "Write code to read and process JSON in Python."
    ]
]

def save_prompt_pairs(prompt_pairs):
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, "prompt_pairs.json"), "w") as f:
        json.dump(prompt_pairs, f, indent=2)
    print(f"Saved {len(prompt_pairs)} prompt pairs to {SAVE_DIR}/prompt_pairs.json")
    
class SimpleSparseSAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, tie_weights=False, dtype=DTYPE):
        super().__init__()
        self.encoder = torch.nn.Linear(input_dim, latent_dim, dtype=dtype)
        if tie_weights:
            self.decoder = torch.nn.Linear(latent_dim, input_dim, dtype=dtype)
            self.decoder.weight = torch.nn.Parameter(self.encoder.weight.T)
        else:
            self.decoder = torch.nn.Linear(latent_dim, input_dim, dtype=dtype)
        
    def forward(self, x):
        if x.dtype != self.encoder.weight.dtype:
            x = x.to(self.encoder.weight.dtype)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=DEVICE))

def load_model_and_tokenizer():
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=DTYPE,
        device_map=DEVICE,
        output_hidden_states=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

def extract_hidden_states(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states[LAYER_IDX][0] 
    return hidden_states, inputs, outputs

def train_sae(model, tokenizer, train_prompts, input_dim, latent_dim=SAE_DIM):
    print("Training Sparse Autoencoder...")
    sae_model = SimpleSparseSAE(input_dim, latent_dim, dtype=DTYPE).to(DEVICE)
    
    optimizer = torch.optim.Adam(sae_model.parameters(), lr=1e-3)
    reconstruction_loss_fn = torch.nn.MSELoss()
    sparsity_loss_weight = 0.1
    
    sae_model.train()
    for epoch in range(3):  
        total_loss = 0
        for prompt in tqdm(train_prompts):
            hidden_states, _, _ = extract_hidden_states(model, tokenizer, prompt)
            
            optimizer.zero_grad()
            encoded, decoded = sae_model(hidden_states)
            
            recon_loss = reconstruction_loss_fn(decoded, hidden_states)
            sparsity_loss = torch.mean(torch.abs(encoded)) 
            loss = recon_loss + sparsity_loss_weight * sparsity_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    sae_model.save(os.path.join(SAVE_DIR, "sae_model.pt"))
    print(f"Saved SAE model to {SAVE_DIR}/sae_model.pt")
    return sae_model

def compute_gradients(model, tokenizer, sae_model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_state = outputs.hidden_states[LAYER_IDX][0].detach()
    
    hidden_state = hidden_state.clone().detach().requires_grad_(True)
    
    sae_model.eval()
    encoded, _ = sae_model(hidden_state)  
    
    predicted_token_id = outputs.logits[0, -1].argmax().item()
    
    # Creating a gradient by directly taking the gradient of the sum of SAE features w.r.t each feature
    gradient_target = encoded.sum()
    gradient_target.backward()
    
    feature_gradients = hidden_state.grad[-1]  
    
    return feature_gradients, predicted_token_id

def alternative_compute_gradients(model, tokenizer, sae_model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_state = outputs.hidden_states[LAYER_IDX][0]
        
        encoded, _ = sae_model(hidden_state)
        
        predicted_token_id = outputs.logits[0, -1].argmax().item()
        
        feature_importance = torch.abs(encoded[-1])
        
    return feature_importance, predicted_token_id

def compare_features(gradients1, gradients2):
    top_k1 = set(torch.topk(torch.abs(gradients1), TOP_K).indices.cpu().numpy())
    top_k2 = set(torch.topk(torch.abs(gradients2), TOP_K).indices.cpu().numpy())
    
    jaccard = len(top_k1.intersection(top_k2)) / len(top_k1.union(top_k2))
    
    vec1 = gradients1.cpu().detach().numpy().flatten()
    vec2 = gradients2.cpu().detach().numpy().flatten()
    
    cos_sim = cosine_similarity(
        vec1.reshape(1, -1),
        vec2.reshape(1, -1)
    )[0][0]
    
    return {
        "jaccard_similarity": float(jaccard),  
        "cosine_similarity": float(cos_sim),
        "top_k_intersection": [int(x) for x in list(top_k1.intersection(top_k2))],
        "top_k_unique_1": [int(x) for x in list(top_k1 - top_k2)],
        "top_k_unique_2": [int(x) for x in list(top_k2 - top_k1)],
    }

def analyze_prompt_pairs(model, tokenizer, sae_model, prompt_pairs):
    """Analyze the feature overlap between prompt pairs"""
    results = []
    
    for i, (prompt1, prompt2) in enumerate(prompt_pairs):
        print(f"Analyzing prompt pair {i+1}/{len(prompt_pairs)}")
        
        gradients1, token_id1 = alternative_compute_gradients(model, tokenizer, sae_model, prompt1)
        gradients2, token_id2 = alternative_compute_gradients(model, tokenizer, sae_model, prompt2)
        
        comparison = compare_features(gradients1, gradients2)
        
        results.append({
            "prompt1": prompt1,
            "prompt2": prompt2,
            "predicted_token1": tokenizer.decode([token_id1]),
            "predicted_token2": tokenizer.decode([token_id2]),
            "comparison": comparison
        })
    
    with open(os.path.join(SAVE_DIR, "analysis_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved analysis results to {SAVE_DIR}/analysis_results.json")
    return results

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    save_prompt_pairs(prompt_pairs)
    
    model, tokenizer = load_model_and_tokenizer()
    
    input_dim = model.config.hidden_size
    
    training_prompts = [
        "Explain how Python dictionaries work.",
        "What is object-oriented programming?",
        "How do I use regular expressions in Python?",
        "Write a function to calculate the Fibonacci sequence.",
        "Explain the concept of recursion in programming."
    ]
    training_prompts.extend([p for pair in prompt_pairs for p in pair])
    
    sae_path = os.path.join(SAVE_DIR, "sae_model.pt")
    if os.path.exists(sae_path):
        print(f"Loading existing SAE from {sae_path}")
        sae_model = SimpleSparseSAE(input_dim, SAE_DIM, dtype=DTYPE).to(DEVICE)
        sae_model.load(sae_path)
    else:
        sae_model = train_sae(model, tokenizer, training_prompts, input_dim)
    
    results = analyze_prompt_pairs(model, tokenizer, sae_model, prompt_pairs)
    
    jaccard_similarities = [r["comparison"]["jaccard_similarity"] for r in results]
    cosine_similarities = [r["comparison"]["cosine_similarity"] for r in results]
    
    print("\nSummary Statistics:")
    print(f"Average Jaccard Similarity: {np.mean(jaccard_similarities):.4f}")
    print(f"Average Cosine Similarity: {np.mean(cosine_similarities):.4f}")
    print(f"Min Jaccard Similarity: {np.min(jaccard_similarities):.4f}")
    print(f"Max Jaccard Similarity: {np.max(jaccard_similarities):.4f}")

if __name__ == "__main__":
    main()