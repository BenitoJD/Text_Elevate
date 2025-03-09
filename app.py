from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

model_name = "Qwen/Qwen2-0.5B-Instruct"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    print(f"Model {model_name} loaded successfully")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")

rephrase_styles = {
    "standard": "Improve grammar, clarity, and structure while preserving the original meaning.",
    "formal": "Make the text more formal, professional, and sophisticated.",
    "casual": "Make the text more casual, conversational, and friendly.",
    "concise": "Make the text more concise and to the point.",
    "elaborate": "Expand the text with more details and explanations.",
    "positive": "Rephrase to have a more positive and optimistic tone.",
    "persuasive": "Revise the text to be more convincing and compelling, emphasizing key points to influence the reader.",
    "technical": "Adjust the text to include precise, industry-specific terminology and a structured, factual tone suitable for technical audiences.",
    "diplomatic": "Soften the tone and refine the language to be tactful, neutral, and respectful, ideal for sensitive or collaborative situations.",
    "authoritative": "Strengthen the text with a confident, commanding tone that conveys expertise and leadership.",
    "action-oriented": "Rephrase the text to focus on clear, actionable steps or outcomes, emphasizing practicality and results.",
    "inclusive": "Modify the text to be welcoming and considerate of diverse perspectives, fostering a sense of collaboration and equity.",
    "professional-email": "Refine the text into a polished, concise, and courteous email format suitable for business communication.",
    "urgent-email": "Adjust the text to convey urgency and importance, with a clear call-to-action, while maintaining professionalism.",
    "follow-up-email": "Revise the text into a polite, structured follow-up message that reinforces prior communication and prompts a response.",
    "informal-email": "Make the text friendly yet professional, striking a balance for a relaxed but respectful email tone.",
    "request-email": "Shape the text into a clear, respectful request with a professional tone, encouraging a positive reply.",
    "update-email": "Transform the text into a succinct, informative update with a professional structure, ideal for status reports or team notifications."
}

@app.route('/')
def index():
    return render_template('index.html', styles=rephrase_styles)

@app.route('/rephrase', methods=['POST'])
def rephrase():
    data = request.json
    original_text = data.get('sentence', '')
    style_key = data.get('style', 'standard')
    
    if not original_text:
        return jsonify({'rephrased': 'Please enter a sentence to rephrase'})
    
    style_instruction = rephrase_styles.get(style_key, rephrase_styles["standard"])
    
    messages = [
        {"role": "system", "content": f"You are a helpful assistant that rephrases sentences. {style_instruction}"},
        {"role": "user", "content": f"Please rephrase this text: \"{original_text}\""}
    ]
    
    if hasattr(tokenizer, 'apply_chat_template'):
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        text = f"System: You are a helpful assistant that rephrases sentences. {style_instruction}\nUser: Please rephrase this text: \"{original_text}\"\nAssistant: "
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            rephrased_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            rephrased_text = rephrased_text.strip().replace('"', '')
        
        response = {
            'rephrased': rephrased_text,
            'style_used': style_key
        }
            
    except Exception as e:
        response = {'rephrased': f"Error rephrasing text: {e}"}
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5010, threaded=True)