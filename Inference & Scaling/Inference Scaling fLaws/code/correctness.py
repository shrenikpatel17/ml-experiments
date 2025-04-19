import json
import re
import time

print("Loading responses from file...")
with open('ai_responses_with_confidence.json', 'r') as f:
    data = json.load(f)

test_cases = {
    "problem_1": [  # Palindrome
        {"input": "racecar", "expected": True},
        {"input": "hello", "expected": False},
        {"input": "", "expected": True},
        {"input": "a", "expected": True},
        {"input": "Racecar", "expected": False},
        {"input": "A man, a plan, a canal: Panama", "expected": False} 
    ],
    "problem_2": [  # Factorial
        {"input": 0, "expected": 1},
        {"input": 1, "expected": 1},
        {"input": 5, "expected": 120},
        {"input": 10, "expected": 3628800},
        {"input": -1, "expected": "error"} 
    ],
    "problem_3": [  # Fibonacci
        {"input": 1, "expected": [0]},
        {"input": 2, "expected": [0, 1]},
        {"input": 5, "expected": [0, 1, 1, 2, 3]},
        {"input": 10, "expected": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]},
        {"input": 0, "expected": []} 
    ],
    "problem_4": [  # Prime number check
        {"input": 2, "expected": True},
        {"input": 7, "expected": True},
        {"input": 10, "expected": False},
        {"input": 1, "expected": False},
        {"input": 0, "expected": False},
        {"input": -5, "expected": False},
        {"input": 997, "expected": True} 
    ],
    "problem_5": [  # Reverse linked list - special case handled separately
        {"input": None, "expected": None}
    ]
}

# Extract code from solution text
def extract_code(solution_text):
    code_block_match = re.search(r'```python\s*(.*?)\s*```', solution_text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1)
    
    any_code_block = re.search(r'```\s*(.*?)\s*```', solution_text, re.DOTALL)
    if any_code_block:
        return any_code_block.group(1)
    
    function_match = re.search(r'(def\s+.*?:.*?)(?=\n\n|\n[^\s]|$)', solution_text, re.DOTALL)
    if function_match:
        return function_match.group(1)
    
    return solution_text

# Test a regular solution
def test_solution(solution_code, test_cases, problem_key):
    namespace = {}
    
    try:
        exec(solution_code, namespace)
        
        function_name = None
        for name, obj in namespace.items():
            if callable(obj) and name not in ['test_solution', 'Node', 'LinkedList']:
                function_name = name
                break
        
        if not function_name:
            return False, "No function found in solution"
        
        results = []
        for case in test_cases:
            try:
                if problem_key == "problem_3":  # Fibonacci
                    result = namespace[function_name](case["input"])
                    if not isinstance(result, list):
                        result = list(result)
                    passed = result == case["expected"]
                    
                elif problem_key == "problem_2" and case["input"] < 0:  # Factorial with negative
                    try:
                        result = namespace[function_name](case["input"])
                        passed = False 
                    except:
                        passed = True
                        result = "error"
                
                else:  
                    result = namespace[function_name](case["input"])
                    passed = result == case["expected"]
                
                results.append({
                    "input": case["input"],
                    "expected": case["expected"],
                    "actual": result,
                    "passed": passed
                })
            except Exception as e:
                if problem_key == "problem_2" and case["input"] < 0 and case["expected"] == "error":
                    passed = True
                else:
                    passed = False
                
                results.append({
                    "input": case["input"],
                    "error": str(e),
                    "passed": passed
                })
        
        all_passed = all(r.get("passed", False) for r in results)
        return all_passed, results
        
    except Exception as e:
        return False, f"Error executing solution: {str(e)}"

# TestS linked list reversal
def test_linked_list_reverse(solution_code):
    namespace = {
        "Node": type("Node", (), {"__init__": lambda self, data, next=None: setattr(self, "data", data) or setattr(self, "next", next)}),
    }
    
    try:
        exec(solution_code, namespace)
        
        function_name = None
        for name, obj in namespace.items():
            if callable(obj) and name not in ['Node', 'LinkedList', 'test_linked_list_reverse']:
                function_name = name
                break
        
        if not function_name:
            for name, obj in namespace.items():
                if isinstance(obj, type) and hasattr(obj, 'reverse'):
                    function_name = f"{name}.reverse"
                    break
        
        if not function_name:
            return False, "No reverse function or method found"
        
        test_cases = [
            {"input": [1, 2, 3, 4, 5], "expected": [5, 4, 3, 2, 1]},
            {"input": [1], "expected": [1]},
            {"input": [], "expected": []}
        ]
        
        results = []
        
        for case in test_cases:
            try:
                if not case["input"]:
                    head = None
                else:
                    head = namespace["Node"](case["input"][0])
                    current = head
                    for val in case["input"][1:]:
                        current.next = namespace["Node"](val)
                        current = current.next
                
                if "." in function_name:
                    class_name, method_name = function_name.split(".")
                    if hasattr(namespace[class_name], "reverse"):
                        if hasattr(namespace[class_name].reverse, "__self__"):
                            reversed_head = getattr(namespace[class_name], method_name)(head)
                        else:
                            instance = namespace[class_name]()
                            reversed_head = getattr(instance, method_name)(head)
                    else:
                        reversed_head = None
                else:
                    reversed_head = namespace[function_name](head)
                
                result = []
                current = reversed_head
                while current:
                    result.append(current.data)
                    current = current.next
                
                passed = result == case["expected"]
                
                results.append({
                    "input": case["input"],
                    "expected": case["expected"],
                    "actual": result,
                    "passed": passed
                })
                
            except Exception as e:
                results.append({
                    "input": case["input"],
                    "error": str(e),
                    "passed": False
                })
        
        all_passed = all(r.get("passed", False) for r in results)
        return all_passed, results
        
    except Exception as e:
        return False, f"Error executing linked list solution: {str(e)}"

correctness = {}

for problem_key in sorted(data.keys()):
    problem_data = data[problem_key]
    problem_text = problem_data["problem"]
    
    print(f"\nVerifying solutions for {problem_key}: {problem_text}")
    
    problem_correctness = []
    
    for i, result in enumerate(problem_data["results"]):
        print(f"\nTesting solution {i+1}...")
        
        solution_text = result["response"]
        solution_code = extract_code(solution_text)
        
        if problem_key == "problem_5":  
            is_correct, test_details = test_linked_list_reverse(solution_code)
        else:
            is_correct, test_details = test_solution(solution_code, test_cases[problem_key], problem_key)
        
        print(f"Solution {i+1}: {'CORRECT' if is_correct else 'INCORRECT'}")
        problem_correctness.append(is_correct)
        
        if not is_correct and isinstance(test_details, list):
            for test in test_details:
                if not test.get("passed", False):
                    print(f"  Failed: input={test['input']}")
                    print(f"          expected={test.get('expected')}")
                    if 'error' in test:
                        print(f"          error={test.get('error')}")
                    else:
                        print(f"          actual={test.get('actual', 'N/A')}")
        
        elif not is_correct and isinstance(test_details, str):
            print(f"  Error: {test_details}")
            
        time.sleep(0.1)
    
    correctness[problem_key] = problem_correctness
    print(f"\nResults for {problem_key}: {problem_correctness}")

with open('solution_correctness.json', 'w') as f:
    json.dump(correctness, f, indent=2)

print("\nAll solution correctness results saved to solution_correctness.json")

# Update the original analysis script with these correctness results
analysis_script = """
import json
import matplotlib.pyplot as plt
import numpy as np

# Load the results with confidence scores
with open('ai_responses_with_confidence.json', 'r') as f:
    data = json.load(f)

# Load the verified correctness data
with open('solution_correctness.json', 'r') as f:
    correctness = json.load(f)

# Analyze the relationship between confidence and correctness
all_confidences = []
all_correctness = []
false_positives = []  # High confidence but wrong
false_negatives = []  # Low confidence but right
true_positives = []   # High confidence and right
true_negatives = []   # Low confidence and wrong

# Define confidence threshold
confidence_threshold = 0.7

for problem_key, problem_data in data.items():
    problem_results = problem_data["results"]
    
    for i, result in enumerate(problem_results):
        if "confidence" in result and "avg_probability" in result["confidence"]:
            confidence = result["confidence"]["avg_probability"]
            correct = correctness[problem_key][i]
            
            all_confidences.append(confidence)
            all_correctness.append(1 if correct else 0)
            
            # Categorize results
            if confidence > confidence_threshold and not correct:
                false_positives.append((confidence, problem_key, i))
            elif confidence <= confidence_threshold and correct:
                false_negatives.append((confidence, problem_key, i))
            elif confidence > confidence_threshold and correct:
                true_positives.append((confidence, problem_key, i))
            else:
                true_negatives.append((confidence, problem_key, i))

# Print statistics
print(f"Total responses analyzed: {len(all_confidences)}")
print(f"Average confidence: {np.mean(all_confidences):.4f}")
print(f"Accuracy: {np.mean(all_correctness):.4f}")
print(f"False positives (overconfident errors): {len(false_positives)}")
print(f"False negatives (underconfident correct): {len(false_negatives)}")
print(f"True positives (confident correct): {len(true_positives)}")
print(f"True negatives (unconfident errors): {len(true_negatives)}")

# Plot confidence vs correctness
plt.figure(figsize=(10, 6))
plt.scatter(all_confidences, all_correctness, alpha=0.7)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axvline(x=confidence_threshold, color='r', linestyle='--')
plt.xlabel('Confidence Score')
plt.ylabel('Correctness (1 = Correct, 0 = Incorrect)')
plt.title('AI Confidence vs Correctness')
plt.grid(True)
plt.savefig('confidence_vs_correctness.png')
print("Analysis plot saved as confidence_vs_correctness.png")

# Report on most interesting cases
print("\\nTop 3 False Positives (most overconfident errors):")
for confidence, problem_key, index in sorted(false_positives, reverse=True)[:3]:
    print(f"Problem: {data[problem_key]['problem']}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Response excerpt: {data[problem_key]['results'][index]['response'][:100]}...\\n")

print("\\nTop 3 False Negatives (most underconfident correct answers):")
for confidence, problem_key, index in sorted(false_negatives)[:3]:
    print(f"Problem: {data[problem_key]['problem']}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Response excerpt: {data[problem_key]['results'][index]['response'][:100]}...\\n")
"""

# Save the analysis script
with open('analyze_results.py', 'w') as f:
    f.write(analysis_script)

print("\nAnalysis script saved to analyze_results.py")
print("Run this script after verifying solutions to analyze confidence vs correctness")