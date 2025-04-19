import json
import matplotlib.pyplot as plt
import numpy as np

with open('ai_responses_with_confidence.json', 'r') as f:
    data = json.load(f)

correctness = {
    "problem_1": [True, True, True, True, True],  
    "problem_2": [True, True, True, True, True],  
    "problem_3": [False, True, False, True, False],  
    "problem_4": [True, True, True, True, True],  
    "problem_5": [False, False, False, False, True]  
}

all_confidences = []
all_correctness = []
false_positives = []
false_negatives = []
true_positives = []
true_negatives = []

confidence_threshold = 0.8

for problem_key, problem_data in data.items():
    problem_results = problem_data["results"]
    
    for i, result in enumerate(problem_results):
        if "confidence" in result and "avg_probability" in result["confidence"]:
            confidence = result["confidence"]["avg_probability"]
            correct = correctness[problem_key][i]
            
            all_confidences.append(confidence)
            all_correctness.append(1 if correct else 0)
            
            if confidence > confidence_threshold and not correct:
                false_positives.append((confidence, problem_key, i))
            elif confidence <= confidence_threshold and correct:
                false_negatives.append((confidence, problem_key, i))
            elif confidence > confidence_threshold and correct:
                true_positives.append((confidence, problem_key, i))
            else:
                true_negatives.append((confidence, problem_key, i))

print(f"Total responses analyzed: {len(all_confidences)}")
print(f"Average confidence: {np.mean(all_confidences):.4f}")
print(f"Accuracy: {np.mean(all_correctness):.4f}")
print(f"False positives (overconfident errors): {len(false_positives)}")
print(f"False negatives (underconfident correct): {len(false_negatives)}")
print(f"True positives (confident correct): {len(true_positives)}")
print(f"True negatives (unconfident errors): {len(true_negatives)}")

print("\nTop 3 False Positives (most overconfident errors):")
for confidence, problem_key, index in sorted(false_positives, reverse=True)[:3]:
    print(f"Problem: {data[problem_key]['problem']}")
    print(f"Confidence: {confidence:.4f}")

print("\nTop 3 False Negatives (most underconfident correct answers):")
for confidence, problem_key, index in sorted(false_negatives)[:3]:
    print(f"Problem: {data[problem_key]['problem']}")
    print(f"Confidence: {confidence:.4f}")

true_positives_confidence = [conf for conf, _, _ in true_positives]
false_positives_confidence = [conf for conf, _, _ in false_positives]
true_negatives_confidence = [conf for conf, _, _ in true_negatives]
false_negatives_confidence = [conf for conf, _, _ in false_negatives]

# Histogram: Confidence distribution for Correct vs Incorrect answers
plt.figure(figsize=(10, 6))
plt.hist(true_positives_confidence, bins=20, color='g', alpha=0.7, label='True Positives')
plt.hist(false_positives_confidence, bins=20, color='r', alpha=0.7, label='False Positives')
plt.hist(true_negatives_confidence, bins=20, color='b', alpha=0.7, label='True Negatives')
plt.hist(false_negatives_confidence, bins=20, color='orange', alpha=0.7, label='False Negatives')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.title('Confidence Distribution (Correct vs Incorrect)')
ymax = plt.gca().get_ylim()[1]
plt.yticks(range(0, int(ymax) + 1, 1))  
plt.legend(loc='upper left', bbox_to_anchor=(0, 1), borderaxespad=0.)  
plt.grid(True)
plt.tight_layout()
plt.show()

# Scatter Plot: Confidence vs Correctness for each category
plt.figure(figsize=(10, 6))
plt.scatter(true_positives_confidence, [1]*len(true_positives_confidence), color='g', label='True Positives', alpha=0.7)
plt.scatter(false_positives_confidence, [0]*len(false_positives_confidence), color='r', label='False Positives', alpha=0.7)
plt.scatter(true_negatives_confidence, [0]*len(true_negatives_confidence), color='b', label='True Negatives', alpha=0.7)
plt.scatter(false_negatives_confidence, [1]*len(false_negatives_confidence), color='orange', label='False Negatives', alpha=0.7)
plt.axvline(x=confidence_threshold, color='black', linestyle='--', label='Confidence Threshold')
plt.xlabel('Confidence Score')
plt.ylabel('Correctness (1 = Correct, 0 = Incorrect)')
plt.title('Confidence vs Correctness (Categorized)')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.) 
plt.grid(True)
plt.tight_layout()
plt.show()

# Scatter Plot: Confidence vs Correctness, focusing on True Positives & False Positives
plt.figure(figsize=(10, 6))
plt.scatter(true_positives_confidence, [1]*len(true_positives_confidence), color='g', label='True Positives', alpha=0.7)
plt.scatter(false_positives_confidence, [0]*len(false_positives_confidence), color='r', label='False Positives', alpha=0.7)
plt.axvline(x=confidence_threshold, color='black', linestyle='--', label='Confidence Threshold')
plt.xlabel('Confidence Score')
plt.ylabel('Correctness (1 = Correct, 0 = Incorrect)')
plt.title('True Positives vs False Positives')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.) 
plt.grid(True)
plt.tight_layout()
plt.show()

# Scatter Plot: Confidence vs Correctness, focusing on True Negatives & False Negatives
plt.figure(figsize=(10, 6))
plt.scatter(true_negatives_confidence, [0]*len(true_negatives_confidence), color='b', label='True Negatives', alpha=0.7)
plt.scatter(false_negatives_confidence, [1]*len(false_negatives_confidence), color='orange', label='False Negatives', alpha=0.7)
plt.axvline(x=confidence_threshold, color='black', linestyle='--', label='Confidence Threshold')
plt.xlabel('Confidence Score')
plt.ylabel('Correctness (1 = Correct, 0 = Incorrect)')
plt.title('True Negatives vs False Negatives')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.) 
plt.grid(True)
plt.tight_layout()
plt.show()
