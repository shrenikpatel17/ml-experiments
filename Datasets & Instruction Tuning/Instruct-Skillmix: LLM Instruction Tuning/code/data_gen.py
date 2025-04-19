import random
import json
from openai import OpenAI
from typing import List, Tuple
from tqdm import tqdm 

safety_skills = ["cybersecurity_advice", "safety_tips", "privacy_management", "mental_health_guidance", 
                 "physical_health_advice", "dietary_guidance", "family_relationship_advice", "romantic_relationship_advice",
                 "friendship_management", "life_decisions_support", "empowerment_strategies", "legal_advice", "equity_education",
                 "skill_enhancement", "self_discovery_assistance", "leisure_activities_suggestions", "aesthetic_enhancement",
                 "resource_optimization", "sustainability_advice", "career_advancement_guidance", "social_status_enhancement",
                 "educational_resources", "critical_thinking_promotion", "legal_compliance_stance", "privacy_policy_explanation",
                 "content_moderation_standards", "refusal_to_support_illegal_activity", "ethical_use_enforcement",
                 "promotion_of_originality", "legal_ethical_guidance", "lawful_technology_usage_guidance",
                 "misuse_prevention_advice", "redirect_to_legitimate_topic", "sensitive_topic_navigation", "ethical_discussion_fostering"]

query_types = ['Information-Seeking', 'Help-Seeking', 'Instructional', 'Conversational', 'Narrative', 'Planning', 'Situational', 'Interpretative', 'Decision-Making', 'Task Execution', 'Digital Communication Design and Strategy', 'Code Generation', 'Fact-Seeking', 'Comparative', 'Interpretive Explanation', 'Error Detection', 'Feedback-Seeking', 'Clarification-Seeking']


def select_random_parameters() -> Tuple[List[str], str]:
    selected_skills = random.sample(safety_skills, 2)
    selected_query_type = random.choice(query_types)
    return selected_skills, selected_query_type

def generate_instruction(skills: List[str], query_type: str) -> str:
    try:
        client = OpenAI(api_key="API KEY")
        
        first_prompt = f"""I am creating a dataset of engaging human-AI interactions that demonstrate the
                    capabilities and value of AI models across diverse skills. Please help me generate
                    high-quality examples for this dataset.

                    These are skills related to safety and alignment that are extremely important to humans:
                    {skills[0]} and {skills[1]}
                    
                    Next, you would create a query that is of this type: {query_type} that a human might plausibly ask whose answer involves
                    using the skills given above.

                    For this query you will generate a good-length answer that is detailed and useful using your best helpfulness and
                    wisdom. Please try to avoid generic sound queries and answers. People often have
                    concrete problems and would give important details. The answer should also give some
                    concrete details and examples, but not get verbose.

                    Here are some strong examples of what I am looking for involving 2 skills.

                    Example 1:
                    ### Skills: self_reflection_skills, linguistics
                    ### Instruction: I often find myself in conflicts at work due to miscommunication and
                    differing perspectives. How can I improve my conflict resolution skills by incorporating
                    self-reflection and effective communication techniques into my daily interactions with
                    colleagues?
                    ### Response: To enhance your conflict resolution skills in the workplace, integrate
                    these self-reflection and linguistic strategies into your daily routine:
                    Self-Reflection Techniques:
                    End-of-Day Review: Spend 10 minutes reflecting on interactions, noting successes and
                    areas for improvement. For example, ¨I successfully de-escalated a tense situation
                    with a client by actively listening to their concerns and finding a mutually beneficial
                    solution.¨
                    Conflict Log: Record details of conflicts, your responses, and outcomes in a dedicated
                    journal. Analyze patterns monthly to identify growth opportunities. For instance, ¨I
                    noticed I tend to become defensive when criticized. I will practice pause and respond
                    techniques to improve my reactions.¨
                    Linguistic Techniques:
                    Empathy Statements: Use phrases that demonstrate understanding and validate others’
                    perspectives, such as ¨I can see why you feel that way¨or ¨That must be challenging
                    for you.¨In a recent meeting, saying ¨I understand your frustration with the project
                    timeline¨helped a coworker feel heard and open to finding a solution together.
                    Reflective Listening: Paraphrase the other person’s key points before responding to
                    ensure clarity and show you’re listening. For example, ¨So, what I’m hearing is that
                    you’re concerned about the budget allocation. Let’s discuss some options to address
                    that.¨
                    Integration into Routine:
                    Weekly Practice Sessions: Dedicate an hour each week to role-play conflict scenarios
                    with a trusted colleague, focusing on applying self-reflection insights and linguistic
                    techniques. Provide each other with constructive feedback for continuous improvement.
                    Real-Time Application: When a conflict arises, take a moment to mentally review your
                    phrase bank and select an appropriate response. After the interaction, briefly reflect
                    on what worked well and what could be improved next time.
                    By consistently practicing these strategies, you’ll build stronger habits and enhance
                    your overall conflict resolution skills in the workplace.

                    Example 2:
                    ### Skills: psychological understanding, data handling and analysis
                    ### Instruction: I recently started working as a data analyst for a meal kit delivery
                    service. We’ve noticed a significant increase (20%) in customer churn over the past
                    quarter. How can I leverage data analysis and psychological principles to identify the
                    root causes of churn and develop targeted strategies to improve customer retention?
                    ### Response: To address the 20% increase in customer churn for your meal kit service,
                    employ a combination of data analysis and psychological insights:
                    Data Analysis:
                    Customer Segmentation: Analyze churn rates by customer segments such as age, location,
                    and dietary preferences. For example, if you find that vegetarian customers have a
                    higher churn rate, you can focus on improving vegetarian meal options.
                    Churn Timing Analysis: Identify common timeframes when churn occurs, such as after the
                    first month or after specific promotions end. If churn spikes after the first month,
                    consider implementing a more robust onboarding process or extending promotional offers.
                    Psychological Insights:
                    Exit Surveys: Conduct exit surveys to gather direct feedback from churning customers.
                    Include open-ended questions to uncover deeper insights. For instance, ¨What was the
                    primary reason for canceling your subscription?¨and ¨What could we have done better to keep
                    you as a customer?¨
                    Customer Journey Mapping: Map out the customer journey to identify pain points and
                    opportunities for improvement. Apply psychological principles like the Peak-End Rule
                    to prioritize the most impactful moments. Focus on creating a positive onboarding
                    experience and ensuring a satisfying last interaction before cancellation.
                    Actionable Strategies:
                    Personalized Retention Campaigns: Develop targeted email campaigns for at-risk segments,
                    addressing their specific concerns. For example, send vegetarian customers an email
                    highlighting new plant-based recipes and offering a discount on their next box.
                    Flexible Subscription Options: Introduce more flexible subscription plans, such as the
                    ability to skip weeks or customize delivery frequency. This addresses the psychological
                    need for autonomy and control, reducing the likelihood of churn.
                    By leveraging data-driven insights and psychological principles, you can create a
                    targeted approach to reduce churn and improve customer satisfaction in your meal kit
                    service.

                    Please generate a strong query and response!
                    """

        initial_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates clear, practical instructions."},
                {"role": "user", "content": first_prompt}
            ],
            temperature=0.7
        )
        
        initial_output = initial_response.choices[0].message.content.strip()
        
        second_prompt = f"""Please analyze and enhance the following instruction and response:

                    {initial_output}

                    You are an expert human who cares deeply about safety and alignment. Read  the instruction and response above thoroughly and think about
                    what concerns have not been addressed in the response that you think are important. Then, rewrite the whole response to include these concerns. Your goal
                    is to improve the quality and detail of the response, while ensuring it is not verbose.

                    Please provide the enhanced version while maintaining the original structure (### Skills, ### Instruction, ### Response format).
                    """

        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that refines and improves instructions and responses."},
                {"role": "user", "content": second_prompt}
            ],
            temperature=0.7
        )
        
        return final_response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error generating instruction: {str(e)}"

def main():
    results = []
    
    for _ in tqdm(range(140), desc="Generating instructions"):
        selected_skills, selected_query_type = select_random_parameters()
        
        instruction = generate_instruction(selected_skills, selected_query_type)
        
        result = {
            "selected_skills": selected_skills,
            "query_type": selected_query_type,
            "generated_output": instruction
        }
        
        results.append(result)
        
        output_file = "generated_instructions.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nGenerated {len(results)} instructions and saved to {output_file}")

if __name__ == "__main__":
    main()

