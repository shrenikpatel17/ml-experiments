import re
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
import subprocess

SYSTEM_PROMPT = (
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n\n"
    "Problem:"
)

def save_intermediate_results(results, times_taken, detailed_results, temperature, max_new_tokens, model_name, dataset_size):
    with open("experiment_results2.json", "w") as f:
        json.dump({
            "results": {k: float(v) for k, v in results.items()},
            "times_taken": {k: float(v) for k, v in times_taken.items()},
            "detailed_results": {
                k: [
                    {
                        "agreement_ratio": float(r["agreement_ratio"]),
                        "agreement_count": r["agreement_count"],
                        "extracted_answers": r["extracted_answers"],
                        "final_answer": r["final_answer"],
                        "ground_truth": r["ground_truth"],
                        "problem": r["problem"]
                    }
                    for r in results_list
                ]
                for k, results_list in detailed_results.items()
            },
            "settings": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "model": model_name,
                "dataset_size": dataset_size
            }
        }, f, indent=2)

def create_prompt(problem):
    return f"{SYSTEM_PROMPT}\n\n{problem}"

# three different methods to extract the answer from the model's response
def extract_answer(text):
    match = re.search(r"The final answer is (\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1).strip()
    
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]
    
    match = re.search(r"answer is (\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return None

# runs using Ollama llama 3.2 model
def generate_completions(prompt, k=1, temperature=0.7, max_new_tokens=512):
    completions = []
    
    for _ in range(k):
        try:
            cmd = f'ollama run llama3.2 "{prompt}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                completions.append(result.stdout.strip())
            else:
                print(f"Error running Ollama: {result.stderr}")
                completions.append("")
                
        except Exception as e:
            print(f"Error during Ollama call: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            completions.append("")
            
    return completions

def majority_vote(answers):
    vote_dict = {}
    for ans in answers:
        if ans is None:
            continue
        vote_dict[ans] = vote_dict.get(ans, 0) + 1
    if not vote_dict:
        return None
    majority_answer = max(vote_dict.items(), key=lambda x: x[1])[0]
    return majority_answer

def evaluate_example(example, k, temperature):
    problem = example["question"]
    ground_truth = example["answer"].strip()
    ground_truth_number = ground_truth.split("####")[-1].strip()
    
    prompt = create_prompt(problem)
    completions = generate_completions(prompt, k=k, temperature=temperature)
    
    print("\nModel Response:")
    for i, completion in enumerate(completions):
        print(f"Completion {i+1}:")
        print(completion)
        print("-" * 50)
    
    extracted_answers = [extract_answer(text) for text in completions]
    final_answer = majority_vote(extracted_answers)
    
    agreement_with_truth = sum(1 for ans in extracted_answers if ans == ground_truth_number)
    agreement_ratio = agreement_with_truth / k if k > 0 else 0
    
    return {
        "agreement_ratio": agreement_ratio,
        "agreement_count": agreement_with_truth,
        "extracted_answers": extracted_answers,
        "final_answer": final_answer,
        "ground_truth": ground_truth_number,
        "completions": completions,
        "problem": problem
    }

def main():
    dataset = load_dataset("gsm8k", "main", split="test")
    subset = dataset.select(range(10))

    temperature = 1.0 
    max_new_tokens = 1000 
    k_values = [1, 3, 5]
    results = {k: 0 for k in k_values}
    times_taken = {k: 0 for k in k_values}
    
    detailed_results = {k: [] for k in k_values}

    for k in k_values:
        print(f"\nEvaluating with k = {k} completions per problem:")
        total_agreement_ratio = 0
        start_time = time.time()
        for i, example in enumerate(subset):
            try:
                result = evaluate_example(example, k, temperature)
                total_agreement_ratio += result["agreement_ratio"]
                detailed_results[k].append(result)
                
                print(f"Example {i+1}:")
                print("  Ground Truth       :", result["ground_truth"])
                print("  Extracted Answers  :", result["extracted_answers"])
                print("  Final (Majority)   :", result["final_answer"])
                print("  Agreement Count    :", result["agreement_count"], "/", k)
                print("  Agreement Ratio    :", f"{result['agreement_ratio']*100:.2f}%")
                
                elapsed = time.time() - start_time
                current_avg_agreement = total_agreement_ratio / (i + 1)
                results[k] = current_avg_agreement
                times_taken[k] = elapsed
                
                save_intermediate_results(
                    results, 
                    times_taken, 
                    detailed_results, 
                    temperature, 
                    max_new_tokens, 
                    "llama3.2",
                    len(subset)
                )
                print("  Saved intermediate results")
                
            except Exception as e:
                print(f"Error processing example {i+1}: {e}")
                print("Continuing with next example...")
                continue
        
        elapsed = time.time() - start_time
        avg_agreement_ratio = total_agreement_ratio / len(subset)
        results[k] = avg_agreement_ratio
        times_taken[k] = elapsed
        print(f"\nAverage Agreement Ratio with k = {k}: {avg_agreement_ratio*100:.2f}%")
        print(f"Total inference time for k = {k}: {elapsed:.2f} seconds")

    plt.figure(figsize=(15, 10))
    
    # plot 1: agreement ratio vs num of completions
    plt.subplot(2, 2, 1)
    ks = list(results.keys())
    agreement_ratios = [results[k]*100 for k in ks]
    plt.plot(ks, agreement_ratios, marker='o')
    plt.xlabel("Number of Completions (k)")
    plt.ylabel("Average Agreement Ratio (%)")
    plt.title("Agreement Ratio vs Number of Completions")
    plt.grid(True)
    
    # plot 2: inference time vs num of completions
    plt.subplot(2, 2, 2)
    times = [times_taken[k] for k in ks]
    plt.plot(ks, times, marker='o', color='orange')
    plt.xlabel("Number of Completions (k)")
    plt.ylabel("Time (seconds)")
    plt.title("Inference Time vs Number of Completions")
    plt.grid(True)
    
    # plot 3: agreement distribution
    plt.subplot(2, 2, 3)
    agreement_counts = {k: {} for k in k_values}
    for k in k_values:
        for result in detailed_results[k]:
            agreement = result["agreement_count"]
            agreement_counts[k][agreement] = agreement_counts[k].get(agreement, 0) + 1
    
    for k in k_values:
        agreements = list(agreement_counts[k].keys())
        counts = list(agreement_counts[k].values())
        plt.bar([a + k*0.2 for a in agreements], counts, width=0.2, label=f'k={k}')
    plt.xlabel("Number of Agreeing Completions")
    plt.ylabel("Count")
    plt.title("Distribution of Answer Agreement")
    plt.legend()
    plt.grid(True)
    
    # plot 4: per-question agreement ratios
    plt.subplot(2, 2, 4)
    question_agreements = []
    for i in range(len(subset)):
        avg_agreement = sum(detailed_results[k][i]["agreement_ratio"] for k in k_values) / len(k_values)
        question_agreements.append(avg_agreement)
    plt.bar(range(len(question_agreements)), [a*100 for a in question_agreements])
    plt.xlabel("Question Index")
    plt.ylabel("Average Agreement Ratio (%)")
    plt.title("Per-Question Agreement Ratios")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("experiment_analysis2.png")
    plt.show()

    print("\nSummary of Results:")
    for k in k_values:
        print(f"  k = {k}: Average Agreement Ratio = {results[k]*100:.2f}%, Inference time = {times_taken[k]:.2f} seconds")
    
    print("\nResults have been saved to 'experiment_results2.json'")
    print("Analysis plots have been saved to 'experiment_analysis2.png'")

if __name__ == "__main__":
    main()