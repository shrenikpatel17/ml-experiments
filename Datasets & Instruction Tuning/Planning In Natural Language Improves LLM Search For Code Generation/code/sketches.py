import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("API_KEY")) 

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

def generate_observations(problem_description, model="gpt-4", n_observations=5):
    prompt = f"""You are a skilled competitive programmer.
Given the following coding problem, list {n_observations} different high‑level hints or observations (not code) that could help someone solve it.
Each observation should be a unique, plausible approach or insight, stated in clear natural language.

Problem:
\"\"\"
{problem_description.strip()}
\"\"\"

Return each observation on a new line, numbered 1 to {n_observations}."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert problem solver and teacher."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8
    )
    return resp.choices[0].message.content.strip()

if __name__ == "__main__":
    for pid, desc in problems.items():
        print(f"\n=== Observations for {pid} ===")
        obs = generate_observations(desc)
        print(obs)