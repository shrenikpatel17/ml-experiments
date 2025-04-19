import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def query_fine_tuned_model(query, model_path='./fine_tuned_model', max_length=250):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    inputs = tokenizer(query, return_tensors='pt')
    
    outputs = model.generate(
        **inputs, 
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,  
        top_k=50,
        top_p=0.95,
        temperature=0.7 
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

def main():
    # Example queries
    queries = [
        "Instruction: Explain the importance of privacy in data collection.",
        "Instruction: How should a small business handle customer data securely?",
        "Instruction: What are the key components of a comprehensive privacy policy?"
    ]
    
    # Query the model for each example
    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = query_fine_tuned_model(query)
        print("\nResponse:")
        print(response)
        print("-" * 50)

if __name__ == '__main__':
    main()