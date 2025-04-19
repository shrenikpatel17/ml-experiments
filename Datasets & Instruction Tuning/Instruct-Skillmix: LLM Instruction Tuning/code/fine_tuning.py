from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import json
import re

import os

class SafetyAlignmentEvaluation:
    def __init__(self, dataset_path='generated_instructions.json'):
        with open(dataset_path, 'r') as f:
            self.data = json.load(f)
        
        self.original_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.fine_tuned_model = None
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def extract_instruction_and_response(self, generated_output):
        instruction_match = re.search(r'### Instruction: (.*?)### Response:', generated_output, re.DOTALL)
        response_match = re.search(r'### Response: (.*)', generated_output, re.DOTALL)
        
        instruction = instruction_match.group(1).strip() if instruction_match else "No instruction found"
        response = response_match.group(1).strip() if response_match else "No response found"
        
        return instruction, response

    def prepare_training_data(self, output_path='training_data.txt'):
        training_texts = []
        for item in self.data:
            instruction, response = self.extract_instruction_and_response(item['generated_output'])
            
            text = f"Instruction: {instruction}\nResponse: {response}"
            training_texts.append(text)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(training_texts))
        
        return output_path

    def fine_tune_model(self, training_file):
        train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=training_file,
            block_size=128
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False
        )
        
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=2,
        )
        
        trainer = Trainer(
            model=self.original_model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )
        
        trainer.train()
        
        self.fine_tuned_model = trainer.model
        self.fine_tuned_model.save_pretrained('./fine_tuned_model')
        self.tokenizer.save_pretrained('./fine_tuned_model')

    def query_fine_tuned_model(self, query, max_length=200, num_return_sequences=1):
        if self.fine_tuned_model is None:
            if os.path.exists('./fine_tuned_model'):
                self.fine_tuned_model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')
                self.tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')
            else:
                raise ValueError("No fine-tuned model found. Run fine_tune_model() first.")
        
        inputs = self.tokenizer(query, return_tensors='pt')
        
        outputs = self.fine_tuned_model.generate(
            **inputs, 
            max_length=max_length, 
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95
        )
        
        responses = [
            self.tokenizer.decode(output, skip_special_tokens=True) 
            for output in outputs
        ]
        
        return responses

def main():
    evaluator = SafetyAlignmentEvaluation()
    
    training_file = evaluator.prepare_training_data()
    
    evaluator.fine_tune_model(training_file)
    
    # Example of querying the fine-tuned model
    print("Querying fine-tuned model:")
    test_query = "Explain the importance of privacy in data collection"
    responses = evaluator.query_fine_tuned_model(test_query)
    
    for i, response in enumerate(responses, 1):
        print(f"Response {i}:\n{response}\n")

if __name__ == '__main__':
    main()