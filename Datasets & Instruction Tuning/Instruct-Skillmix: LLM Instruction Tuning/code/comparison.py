import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from openai import OpenAI

class ModelComparison:
    def __init__(self, fine_tuned_path='./fine_tuned_model', openai_api_key=None):
        # Original GPT-2 model
        self.original_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.original_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.original_tokenizer.pad_token = self.original_tokenizer.eos_token

        # Fine-tuned model
        self.fine_tuned_model = GPT2LMHeadModel.from_pretrained(fine_tuned_path)
        self.fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(fine_tuned_path)
        self.fine_tuned_tokenizer.pad_token = self.fine_tuned_tokenizer.eos_token

        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            self.client = None

    def generate_response(self, query, model, tokenizer, max_length=750):
        inputs = tokenizer(query, return_tensors='pt')
        
        outputs = model.generate(
            **inputs, 
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_gpt4o_response(self, query, max_length=750):
        try:
            if not self.client:
                raise ValueError("OpenAI API key not set")
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query}
                ],
                max_tokens=max_length
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating GPT-4o response: {e}")
            return ""

    def compare_extended_models(self, queries, openai_api_key=None):
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)

        comparison_results = []

        for i, query in enumerate(queries):
            max_length = 250 if i < 5 else 500

            gpt2_response = self.generate_response(query, self.original_model, self.original_tokenizer, max_length)
            fine_tuned_response = self.generate_response(query, self.fine_tuned_model, self.fine_tuned_tokenizer, max_length)
            
            gpt4o_response = self.generate_gpt4o_response(query, max_length)

            comparison_results.append({
                'query': query,
                'gpt2_response': gpt2_response,
                'fine_tuned_response': fine_tuned_response,
                'gpt4o_response': gpt4o_response,
                'max_length': max_length
            })

        return comparison_results

def main():
    queries = [
        "As a developer building an AI chatbot for customer support, I want to ensure that my model does not generate harmful, biased, or misleading responses. Could you help me implement safety measures and alignment techniques to minimize risks?",
        "As a cybersecurity consultant, I need to educate small businesses on best practices for protecting customer data from breaches and cyberattacks. Could you provide a comprehensive guide on security measures and compliance requirements?",
        "As a teacher integrating AI-driven tools into my classroom, I am concerned about student privacy and algorithmic biases. How can I ensure ethical AI use while complying with data protection laws in an educational setting?",
        "As a researcher developing an AI model for medical diagnostics, I want to mitigate ethical concerns regarding data privacy, bias in predictions, and patient consent. Can you help me design an AI ethics framework tailored to healthcare applications?",
        "As a social media platform owner, I want to prevent misinformation, harmful content, and manipulation while preserving free speech. How can I align my content moderation policies with ethical AI principles?",
        "As a policymaker working on AI regulation, I need to understand the risks of algorithmic decision-making in areas like hiring, lending, and law enforcement. Could you provide insights on how to balance fairness, accountability, and efficiency?",
        "As an entrepreneur developing a blockchain-based identity verification system, I want to ensure that my solution enhances security and user privacy while aligning with global regulations. Could you outline the key considerations for ethical implementation?",
        "As an AI ethics researcher, I am investigating the impact of automated surveillance technologies on civil liberties. Can you provide a critical analysis of the risks, benefits, and policy recommendations to ensure responsible use?",
        "As a smart home device manufacturer, I am concerned about user privacy and potential security vulnerabilities in AI-powered assistants. How can I design safer, more transparent systems that align with ethical best practices?",
        "As a journalist covering AI safety and alignment, I want to educate the public on the risks of unaligned AI systems and potential solutions. Could you help me develop an accessible and well-researched article on this topic?",
    ]

    comparison = ModelComparison(openai_api_key="API KEY")

    try:
        results = comparison.compare_extended_models(queries)

        output_file = 'model_comparison_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Comparison results saved to {output_file}")

        for result in results:
            print(f"\nQuery: {result['query']}")
            print(f"Max Length: {result['max_length']} words")
            print(f"GPT-2 Response Length: {len(result['gpt2_response'].split())}")
            print(f"Fine-Tuned Response Length: {len(result['fine_tuned_response'].split())}")
            print(f"GPT-4o Response Length: {len(result['gpt4o_response'].split())}")

    except Exception as e:
        print(f"Error in model comparison: {e}")

if __name__ == '__main__':
    main()