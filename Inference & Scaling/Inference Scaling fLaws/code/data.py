import openai
import json
import time

# Set your OpenAI API key
openai.api_key = "API KEY"  

problems = [
    "Write a function to check if a string is a palindrome.",
    "Write a function to find the factorial of a number.",
    "Write a function to calculate the Fibonacci sequence up to n terms.",
    "Write a function to check if a number is prime.",
    "Write a function to reverse a linked list."
]

results = {}

for i, problem in enumerate(problems):
    print(f"Processing problem {i+1}: {problem}")
    
    responses = []
    
    for j in range(5):
        try:
            print(f"  Getting response {j+1}...")
            
            # Make API call to GPT-4
            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": problem}]
            )
            
            answer = completion.choices[0].message.content
            responses.append(answer)
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Error: {e}")
            responses.append(f"Error: {str(e)}")
    
    results[f"problem_{i+1}"] = {
        "problem": problem,
        "responses": responses
    }

with open('ai_responses.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Responses saved to ai_responses.json")