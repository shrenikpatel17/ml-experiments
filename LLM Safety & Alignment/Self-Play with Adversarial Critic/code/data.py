from openai import OpenAI
import random
import json
import os
import re
import time
import string

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "API KEY"))

# Normalizes text for comaprison
def normalize_text(text):
    translator = str.maketrans('', '', string.punctuation)
    normalized = text.translate(translator).lower()
    normalized = ' '.join(normalized.split())
    return normalized

# Checks if a prompt is semantically similar to existing ones
def is_similar_prompt(new_prompt, existing_prompts, similarity_threshold=0.7):

    normalized_new = normalize_text(new_prompt)
    
    if normalized_new in existing_prompts:
        return True
    
    return False

# Loads all existing prompts from files
def load_existing_prompts():
    existing_prompts = set()
    
    if os.path.exists("synthetic_dataset_progress.json"):
        try:
            with open("synthetic_dataset_progress.json", "r") as f:
                progress_data = json.load(f)
                for item in progress_data:
                    existing_prompts.add(normalize_text(item["prompt"]))
                print(f"Loaded {len(progress_data)} prompts from progress file.")
        except Exception as e:
            print(f"Error loading progress file: {e}")
    
    if os.path.exists("synthetic_dataset_full.json"):
        try:
            with open("synthetic_dataset_full.json", "r") as f:
                full_data = json.load(f)
                for item in full_data:
                    existing_prompts.add(normalize_text(item["prompt"]))
                print(f"Loaded {len(full_data)} prompts from full dataset file.")
        except Exception as e:
            print(f"Error loading full dataset file: {e}")
    
    batch_files = [f for f in os.listdir('.') if f.startswith('synthetic_dataset_batch_')]
    for batch_file in batch_files:
        try:
            with open(batch_file, "r") as f:
                batch_data = json.load(f)
                for item in batch_data:
                    existing_prompts.add(normalize_text(item["prompt"]))
                print(f"Loaded {len(batch_data)} prompts from {batch_file}.")
        except Exception as e:
            print(f"Error loading batch file {batch_file}: {e}")
    
    print(f"Total unique prompts already generated: {len(existing_prompts)}")
    return existing_prompts

# Generates unique prompts using OpenAI API
def generate_unique_prompts(n=10, existing_prompts=None, max_attempts=5):
    if existing_prompts is None:
        existing_prompts = set()
    
    unique_batch = []
    attempts = 0
    
    while len(unique_batch) < n and attempts < max_attempts:
        attempts += 1
        remaining = n - len(unique_batch)
        request_size = min(remaining * 3, 60)
        system_prompt = f"""Generate {request_size} diverse and unique questions or prompts that a language model might respond to.
        
Requirements:
- Include a mix of factual questions, how-to requests, and opinion questions
- Each prompt should be clear and concise (10-15 words max)
- Cover diverse topics (e.g., science, history, everyday life, technology)
- Avoid controversial, political, or sensitive topics
- Each prompt should be on a new line with no numbering
- Each prompt MUST be completely unique and different from others
- Use varied question formats and structures
- Explore diverse subject areas including:
  * Science and nature
  * Technology and computing
  * Arts and culture
  * Food and cooking
  * Health and fitness
  * Travel and geography
  * Business and finance
  * Educational concepts
  * Everyday practical advice
  * History and social studies

Examples of good prompts:
"What is the capital of Brazil?"
"How do I bake a chocolate cake from scratch?"
"What causes rainbows to appear after rain?"
"Can you explain how photosynthesis works?"
"What are some good exercises for beginners?"
"How much sleep does an average adult need?"

YOUR PROMPTS MUST BE COMPLETELY DIFFERENT FROM THESE EXAMPLES.
Use these only as format references.

Important: Generate completely UNIQUE prompts unlike any others.
"""

        print(f"Attempt {attempts}/{max_attempts}: Requesting {request_size} candidate prompts...")
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": system_prompt}],
            max_tokens=1500,
            temperature=0.9
        )
        
        generated_prompts = response.choices[0].message.content.strip().split("\n")
        cleaned_prompts = [p.strip().strip('"').strip("'") for p in generated_prompts if p.strip()]
        
        added_count = 0
        for prompt in cleaned_prompts:
            if prompt and not is_similar_prompt(prompt, existing_prompts) and len(unique_batch) < n:
                normalized = normalize_text(prompt)
                existing_prompts.add(normalized)
                unique_batch.append(prompt)
                added_count += 1
        
        print(f"Added {added_count} unique prompts in this attempt. Total: {len(unique_batch)}/{n}")
        
        if len(unique_batch) < n and attempts < max_attempts:
            time.sleep(2)
    
    return unique_batch

# Gets responses for a single prompt
def get_responses_for_prompt(prompt, is_high_coverage=True, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            if is_high_coverage:
                system_prompt = f"""I'll give you a prompt. Please provide two responses:
1. A helpful, accurate, and informative response.
2. An unhelpful, vague, or incorrect response.

The prompt is: "{prompt}"

Response 1 should be genuinely useful and accurate.
Response 2 should be clearly worse - either misleading, incomplete, or inaccurate.

Format your answer exactly like this:
RESPONSE 1: [helpful response text]
RESPONSE 2: [unhelpful response text]

Do not include any other text in your response.
"""
                preferred = 0  
            else:
                system_prompt = f"""I'll give you a prompt. Please provide two different but equally valid responses.

The prompt is: "{prompt}"

Both responses should be helpful but take different approaches or emphasize different aspects.
They should be of similar quality but distinctive in content or perspective.

Format your answer exactly like this:
RESPONSE 1: [first response text]
RESPONSE 2: [second response text]

Do not include any other text in your response.
"""
                preferred = random.choice([0, 1])  
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "system", "content": system_prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse the responses
            response1_match = re.search(r'RESPONSE 1:(.*?)(?=RESPONSE 2:|$)', content, re.DOTALL)
            response2_match = re.search(r'RESPONSE 2:(.*?)(?=$)', content, re.DOTALL)
            
            response1 = response1_match.group(1).strip() if response1_match else None
            response2 = response2_match.group(1).strip() if response2_match else None
            
            if response1 and response2:
                return response1, response2, preferred
            
            retries += 1
            print(f"Retrying prompt ({retries}/{max_retries}): {prompt} - Got incomplete responses")
            time.sleep(1) 
            
        except Exception as e:
            retries += 1
            print(f"Error getting responses ({retries}/{max_retries}): {e}")
            time.sleep(2) 
    
    return (
        f"A helpful response about {prompt}",
        f"A less helpful response about {prompt}",
        0 if is_high_coverage else random.choice([0, 1])
    )

def create_unique_dataset(n=500, batch_size=10, save_interval=10):
    print(f"Generating dataset with {n} unique examples...")
    
    dataset = []
    if os.path.exists("synthetic_dataset_progress.json"):
        try:
            with open("synthetic_dataset_progress.json", "r") as f:
                dataset = json.load(f)
                print(f"Loaded {len(dataset)} existing examples from progress file.")
        except Exception as e:
            print(f"Error loading existing dataset: {e}")
    
    existing_prompts = load_existing_prompts()
    
    remaining = n - len(dataset)
    batch_num = 1
    
    while remaining > 0:
        current_batch_size = min(batch_size, remaining)
        print(f"\n--- Batch {batch_num} ---")
        print(f"Generating batch of {current_batch_size} unique prompts ({n - remaining + 1}-{n - remaining + current_batch_size} of {n})...")
        
        prompts = generate_unique_prompts(current_batch_size, existing_prompts)
        
        if len(prompts) == 0:
            print("Could not generate any more unique prompts in this batch. Waiting before retry...")
            time.sleep(5) 
            continue
        
        batch_data = []
        coverage = ["high" if random.random() < 0.5 else "low" for _ in range(len(prompts))]
        
        for i, prompt in enumerate(prompts):
            is_high_coverage = coverage[i] == "high"
            coverage_type = "high" if is_high_coverage else "low"
            current_index = n - remaining + i + 1
            print(f"Processing prompt {current_index} of {n} ({coverage_type} coverage): {prompt}")
            
            try:
                response1, response2, preferred = get_responses_for_prompt(prompt, is_high_coverage)
                
                example = {
                    "prompt": prompt,
                    "response1": response1,
                    "response2": response2,
                    "preferred": preferred
                }
                
                batch_data.append(example)
                dataset.append(example)
                
                print(f"Successfully processed. Preferred: {preferred}")
                
                if current_index % save_interval == 0:
                    with open("synthetic_dataset_progress.json", "w") as f:
                        json.dump(dataset, f, indent=2)
                    print(f"Progress saved to 'synthetic_dataset_progress.json' ({len(dataset)} examples)")
                
            except Exception as e:
                print(f"Error processing prompt {current_index}: {e}")
                example = {
                    "prompt": prompt,
                    "response1": f"A helpful response about {prompt}",
                    "response2": f"A less helpful response about {prompt}",
                    "preferred": 0 if is_high_coverage else random.choice([0, 1])
                }
                batch_data.append(example)
                dataset.append(example)
        
        with open(f"synthetic_dataset_batch_{batch_num}.json", "w") as f:
            json.dump(batch_data, f, indent=2)
        print(f"Batch {batch_num} saved to 'synthetic_dataset_batch_{batch_num}.json' ({len(batch_data)} examples)")
        
        remaining = n - len(dataset)
        batch_num += 1
    
    random.shuffle(dataset)
    return dataset

if __name__ == "__main__":
    try:
        full_dataset = create_unique_dataset(500, batch_size=10, save_interval=10)
        
        with open("synthetic_dataset_full.json", "w") as f:
            json.dump(full_dataset, f, indent=2)
        
        print(f"\nFull dataset with {len(full_dataset)} unique examples saved to 'synthetic_dataset_full.json'.")
        
        prompts = [normalize_text(item["prompt"]) for item in full_dataset]
        unique_prompts = set(prompts)
        
        print(f"Uniqueness check: {len(unique_prompts)} unique prompts out of {len(prompts)} total.")
        if len(unique_prompts) < len(prompts):
            print("WARNING: There are still duplicate prompts in the dataset.")
            seen = {}
            duplicates = []
            for i, p in enumerate(prompts):
                if p in seen:
                    duplicates.append((p, seen[p], i))
                else:
                    seen[p] = i
            
            print(f"Found {len(duplicates)} duplicates:")
            for dup, first_idx, second_idx in duplicates[:10]:  # Show first 10 duplicates
                print(f"  - '{full_dataset[first_idx]['prompt']}' (indices {first_idx}, {second_idx})")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()