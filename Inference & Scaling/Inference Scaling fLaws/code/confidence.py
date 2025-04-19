import openai
import json
import time
import numpy as np
import math

openai.api_key = "API KEY"  # Replace with your actual API key

with open('ai_responses.json', 'r') as f:
    results = json.load(f)

confidence_data = {}

for problem_key, problem_data in results.items():
    problem = problem_data["problem"]
    responses = problem_data["responses"]
    
    problem_confidence = []
    
    print(f"Processing confidence for {problem_key}")
    
    for i, response in enumerate(responses):
        try:
            print(f"  Calculating confidence for response {i+1}...")
            
            chat_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": response}
                ],
                logprobs=True,
                top_logprobs=5  # Request top 5 log probabilities per token
            )
            
            logprobs_data = chat_response.choices[0].logprobs
            
            tokens = []
            token_logprobs = []
            
            if hasattr(logprobs_data, 'content'):
                for token_info in logprobs_data.content:
                    if token_info.token:
                        tokens.append(token_info.token)
                        token_logprobs.append(token_info.logprob)
            
            # Calculate confidence metrics
            if token_logprobs:
                probabilities = [math.exp(lp) for lp in token_logprobs]
                
                avg_prob = np.mean(probabilities)
                min_prob = min(probabilities)
                max_prob = max(probabilities)
                
                problem_confidence.append({
                    "response_index": i,
                    "avg_probability": float(avg_prob),  # Convert numpy types to native Python for JSON
                    "min_probability": float(min_prob),
                    "max_probability": float(max_prob),
                    "avg_logprob": float(np.mean(token_logprobs)),
                    "token_count": len(tokens)
                })
            else:
                problem_confidence.append({
                    "response_index": i,
                    "error": "No logprobs data available",
                    "avg_probability": 0,
                    "min_probability": 0,
                    "max_probability": 0
                })
            
            time.sleep(2)
            
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            problem_confidence.append({
                "response_index": i,
                "error": str(e),
                "avg_probability": 0,
                "min_probability": 0,
                "max_probability": 0
            })
    
    confidence_data[problem_key] = problem_confidence

# Save confidence scores to JSON file
with open('ai_confidence_scores.json', 'w') as f:
    json.dump(confidence_data, f, indent=2)

print("Confidence scores saved to ai_confidence_scores.json")

combined_results = {}

for problem_key, problem_data in results.items():
    combined_results[problem_key] = {
        "problem": problem_data["problem"],
        "results": []
    }
    
    for i, response in enumerate(problem_data["responses"]):
        confidence = next((c for c in confidence_data.get(problem_key, []) 
                          if c.get("response_index") == i), {})
        
        combined_results[problem_key]["results"].append({
            "response": response,
            "confidence": confidence
        })

with open('ai_responses_with_confidence.json', 'w') as f:
    json.dump(combined_results, f, indent=2)

print("Combined results saved to ai_responses_with_confidence.json")