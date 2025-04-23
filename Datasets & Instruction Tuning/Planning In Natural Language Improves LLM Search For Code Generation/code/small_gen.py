import os
import json
import time
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  

problems = {
    "row_max_score": """
You are given a 0-indexed 2D integer array nums. Initially, your score is 0. Perform the following operations until the matrix becomes empty:

1. From each row in the matrix, select the largest number and remove it. In the case of a tie, it does not matter which number is chosen.
2. Identify the highest number amongst all those removed in step 1. Add that number to your score.

Return the final score.
""",
    "chocolate_rotation": """
You are given a 0-indexed integer array nums of size n representing the cost of collecting different chocolates. The cost of collecting the chocolate at index i is nums[i]. Each chocolate is of a different type, and initially, the chocolate at index i is of i^th type.
In one operation, you can do the following with an incurred cost of x:

- Simultaneously change the chocolate of i^th type to ((i + 1) mod n)^th type for all chocolates.

Return the minimum cost to collect chocolates of all types, given that you can perform as many operations as you would like.
""",
    "binary_inversion": """
You are given a 0-indexed binary string s of length n on which you can apply two types of operations:

1. Choose an index i and invert all characters from index 0 to index i (both inclusive), with a cost of i + 1  
2. Choose an index i and invert all characters from index i to index n - 1 (both inclusive), with a cost of n - i

Return the minimum cost to make all characters of the string equal.
"""
}

test_cases = {
    "row_max_score": [
        {
            "input": [[7,2,1],[6,4,2],[6,5,3],[3,2,1]],
            "expected": 15  
        },
        {
            "input": [[1]],
            "expected": 1  
        }
    ],
    "chocolate_rotation": [
        {
            "input": [20, 1, 15],
            "expected": 13  # Example 1: nums = [20,1,15], x = 5 Output: 13
        },
        {
            "input": [1,2,3],
            "expected": 6  #Example 2: Input: nums = [1,2,3], x = 4 Output: 6
        },
    ],
    "binary_inversion": [
        {
            "input": "0011",
            "expected": 2  
        },
        {
            "input": "010101",
            "expected": 9  
        }
    ]
}

# Create a natural language sketch from observations
def create_sketch_from_observations(observations):
    sketch = "Based on the following observations:\n\n"
    
    for i, obs in enumerate(observations, 1):
        sketch += f"{i}. {obs}\n"
    
    sketch += "\nYou should devise a solution approach that incorporates these insights."
    return sketch

# Generate code based on problem description w/ or w/o a sketch
def generate_code(problem_description, sketch=None, model="gpt-3.5-turbo"):
    if sketch:
        prompt = f"""Solve this coding problem using the provided sketch as guidance:

Problem:
{problem_description.strip()}

Sketch:
{sketch}

Write a complete, efficient Python function to solve this problem. The function should:
1. Have clear inputs and outputs matching the problem statement
2. Be well-commented
3. Handle edge cases
4. Be optimized for performance
5. Use appropriate algorithms and data structures

Your solution should only include the function definition and any helper functions, if needed.
Do not include test cases or code for reading input/output.
"""
    else:
        prompt = f"""Solve this coding problem:

Problem:
{problem_description.strip()}

Write a complete, efficient Python function to solve this problem. The function should:
1. Have clear inputs and outputs matching the problem statement
2. Be well-commented
3. Handle edge cases
4. Be optimized for performance
5. Use appropriate algorithms and data structures

Your solution should only include the function definition and any helper functions, if needed.
Do not include test cases or code for reading input/output.
"""

    retry_count = 0
    max_retries = 3
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert Python programmer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API error: {e}")
            retry_count += 1
            time.sleep(5)  
    
    return "Error: Failed to generate code after multiple attempts."

# Evaluate code against test cases  
def evaluate_code(code_solution, problem_id, test_cases):
    """Evaluate a code solution against test cases."""
    if "```" in code_solution:
        code_lines = code_solution.split("\n")
        clean_lines = []
        in_code_block = False
        
        for line in code_lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            
            if in_code_block:
                clean_lines.append(line)
            
        code_solution = "\n".join(clean_lines)
    
    namespace = {}
    
    try:
        exec(code_solution, namespace)
    except Exception as e:
        return False, f"Compilation error: {str(e)}"
    
    function_name = None
    for line in code_solution.split("\n"):
        if line.strip().startswith("def "):
            function_name = line.strip().split("def ")[1].split("(")[0].strip()
            break
    
    if not function_name or function_name not in namespace:
        return False, "Could not identify function name"
    
    solution_func = namespace[function_name]
    
    for i, test_case in enumerate(test_cases):
        try:
            input_data = test_case["input"]
            expected = test_case["expected"]
            
            result = solution_func(input_data)
            
            if result != expected:
                return False, f"Test case {i+1} failed: expected {expected}, got {result}"
            
        except Exception as e:
            return False, f"Runtime error on test case {i+1}: {str(e)}"
    
    return True, "All test cases passed"

def main():
    with open("observations.json", "r") as f:
        all_observations = json.load(f)
    
    results = {}
    num_attempts = 3  
    
    for problem_id in problems.keys():
        print(f"\n=== Processing {problem_id} ===")
        problem_results = {
            "sketch_attempts": [],
            "baseline_attempts": [],
            "sketch_success_rate": 0.0,
            "baseline_success_rate": 0.0
        }
        
        observations = all_observations[problem_id]
        sketch = create_sketch_from_observations(observations)
        print(f"Created sketch for {problem_id}")
        
        print(f"Generating {num_attempts} code attempts with sketch...")
        sketch_successes = 0
        for attempt in range(num_attempts):
            print(f"  Attempt {attempt + 1}/{num_attempts} with sketch...")
            sketch_code = generate_code(problems[problem_id], sketch)
            if sketch_code:
                passed, message = evaluate_code(sketch_code, problem_id, test_cases[problem_id])
                problem_results["sketch_attempts"].append({
                    "code": sketch_code,
                    "passed": passed,
                    "message": message
                })
                if passed:
                    sketch_successes += 1
                print(f"    {'PASSED' if passed else 'FAILED'}: {message}")
            else:
                print("    Failed to generate valid code")
                problem_results["sketch_attempts"].append({
                    "code": None,
                    "passed": False,
                    "message": "Failed to generate valid code"
                })
            time.sleep(2)  
        
        print(f"Generating {num_attempts} code attempts without sketch...")
        baseline_successes = 0
        for attempt in range(num_attempts):
            print(f"  Attempt {attempt + 1}/{num_attempts} without sketch...")
            baseline_code = generate_code(problems[problem_id])
            if baseline_code:
                passed, message = evaluate_code(baseline_code, problem_id, test_cases[problem_id])
                problem_results["baseline_attempts"].append({
                    "code": baseline_code,
                    "passed": passed,
                    "message": message
                })
                if passed:
                    baseline_successes += 1
                print(f"    {'PASSED' if passed else 'FAILED'}: {message}")
            else:
                print("    Failed to generate valid code")
                problem_results["baseline_attempts"].append({
                    "code": None,
                    "passed": False,
                    "message": "Failed to generate valid code"
                })
            time.sleep(2)  
        
        problem_results["sketch_success_rate"] = sketch_successes / num_attempts
        problem_results["baseline_success_rate"] = baseline_successes / num_attempts
        
        results[problem_id] = problem_results
        
        print(f"\nResults for {problem_id}:")
        print(f"Sketch-based success rate: {problem_results['sketch_success_rate']:.2%} ({sketch_successes}/{num_attempts})")
        print(f"Baseline success rate: {problem_results['baseline_success_rate']:.2%} ({baseline_successes}/{num_attempts})")
    
    with open("sketch_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n=== Overall Results Summary ===")
    total_sketch_successes = sum(1 for problem in results.values() 
                               for attempt in problem["sketch_attempts"] 
                               if attempt["passed"])
    total_baseline_successes = sum(1 for problem in results.values() 
                                 for attempt in problem["baseline_attempts"] 
                                 if attempt["passed"])
    total_attempts = num_attempts * len(problems)
    
    print(f"Overall sketch-based success rate: {total_sketch_successes}/{total_attempts} ({total_sketch_successes/total_attempts:.2%})")
    print(f"Overall baseline success rate: {total_baseline_successes}/{total_attempts} ({total_baseline_successes/total_attempts:.2%})")
    
    report = """# Sketch-Based vs. Baseline Code Generation Results

## Overview
This report compares code solutions generated with guidance from observations (sketch-based) 
to solutions generated directly from problem descriptions (baseline).

## Overall Results
"""
    report += f"- Sketch-based success rate: {total_sketch_successes}/{total_attempts} ({total_sketch_successes/total_attempts:.2%})\n"
    report += f"- Baseline success rate: {total_baseline_successes}/{total_attempts} ({total_baseline_successes/total_attempts:.2%})\n\n"
    
    for problem_id, problem_results in results.items():
        report += f"### {problem_id}\n"
        report += f"- Sketch-based success rate: {problem_results['sketch_success_rate']:.2%}\n"
        report += f"- Baseline success rate: {problem_results['baseline_success_rate']:.2%}\n\n"
        
        report += "#### Sketch\n"
        report += f"```\n{create_sketch_from_observations(all_observations[problem_id])}\n```\n\n"
        
        report += "#### Sketch-Based Solutions\n"
        for i, attempt in enumerate(problem_results["sketch_attempts"], 1):
            report += f"##### Attempt {i} ({'PASSED' if attempt['passed'] else 'FAILED'})\n"
            if attempt["code"]:
                report += f"```python\n{attempt['code']}\n```\n"
                report += f"Result: {attempt['message']}\n\n"
            else:
                report += "Failed to generate valid code\n\n"
        
        report += "#### Baseline Solutions\n"
        for i, attempt in enumerate(problem_results["baseline_attempts"], 1):
            report += f"##### Attempt {i} ({'PASSED' if attempt['passed'] else 'FAILED'})\n"
            if attempt["code"]:
                report += f"```python\n{attempt['code']}\n```\n"
                report += f"Result: {attempt['message']}\n\n"
            else:
                report += "Failed to generate valid code\n\n"
    
    with open("sketch_evaluation_report.md", "w") as f:
        f.write(report)
    
    print("\nResults and report saved successfully!")

if __name__ == "__main__":
    main()