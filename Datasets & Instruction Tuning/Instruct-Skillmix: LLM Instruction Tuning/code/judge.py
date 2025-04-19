from openai import OpenAI

def judge_responses(query, response1, response2, openai_api_key):
    client = OpenAI(api_key=openai_api_key)
    
    system_prompt = """You are an impartial AI judge evaluating two responses to a specific query. Your task is to determine which response is better based on:
    - How well it addresses the specific query
    - Clarity and coherence
    - Depth of explanation
    - Relevance to the topic
    - Informativeness
    - Overall quality

    IMPORTANT:
    - Respond ONLY with 'A', 'B'
    - 'A' means the first response is better
    - 'B' means the second response is better
    - Do not explain your reasoning
    - Be objective and fair in your evaluation"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\n\nResponse A: {response1}\n\nResponse B: {response2}"}
            ],
            max_tokens=10
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error in judging responses: {e}")
        return None

def main():
    openai_api_key = "API KEY"
    
    query = """QUERY"""
    
    responseA = """Response A"""
    
    responseB = """Response B"""
    
    # Get the judgment
    result = judge_responses(query, responseA, responseB, openai_api_key)
    
    if result:
        print(f"\nQuery: {query}")
        print(f"Response A: {responseA}")
        print(f"Response B: {responseB}")
        print(f"\nJudgment: {result}")
        if result == 'A':
            print("Response A is better")
        elif result == 'B':
            print("Response B is better")

if __name__ == "__main__":
    main()
