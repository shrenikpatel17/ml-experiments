from difflib import SequenceMatcher
import re
import json

class RepetitionAnalyzer:
    def __init__(self):
        pass

    def calculate_average_similarity(self, text):
        sentences = [s.strip() for s in re.split('[.!?]+', text) if s.strip()]
        
        if len(sentences) <= 1:
            return 0.0
        
        similarities = []
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                similarity = SequenceMatcher(None, sentences[i], sentences[j]).ratio()
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0

def analyze_responses():
    with open('model_comparison_results.json', 'r') as f:
        results = json.load(f)
    
    analyzer = RepetitionAnalyzer()
    
    gpt2_scores = []
    fine_tuned_scores = []
    
    for i, result in enumerate(results, 1):
        gpt2_score = analyzer.calculate_average_similarity(result['gpt2_response'])
        fine_tuned_score = analyzer.calculate_average_similarity(result['fine_tuned_response'])
        
        gpt2_scores.append(gpt2_score)
        fine_tuned_scores.append(fine_tuned_score)
        
        print(f"\nQuery {i}:")
        print(f"GPT-2 Repetition Score: {gpt2_score:.3f}")
        print(f"Fine-tuned Model Repetition Score: {fine_tuned_score:.3f}")
    
    gpt2_avg = sum(gpt2_scores) / len(gpt2_scores)
    fine_tuned_avg = sum(fine_tuned_scores) / len(fine_tuned_scores)
    
    print("\nFinal Averages:")
    print(f"GPT-2 Average Repetition Score: {gpt2_avg:.3f}")
    print(f"Fine-tuned Model Average Repetition Score: {fine_tuned_avg:.3f}")

if __name__ == "__main__":
    analyze_responses()
