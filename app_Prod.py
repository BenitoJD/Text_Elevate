import os
import logging
from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv
from functools import lru_cache
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

load_dotenv()

sentry_sdk.init(
    dsn=os.getenv('SENTRY_DSN'),
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

class Config:
    MODEL_NAME = os.getenv('MODEL_NAME', 'Qwen/Qwen2-0.5B-Instruct')
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5019))
    MAX_CONTENT_LENGTH = 16 * 1024  # 16KB max request size
    CACHE_SIZE = 128

app.config.from_object(Config)

model = None
tokenizer = None

@lru_cache(maxsize=Config.CACHE_SIZE)
def load_model():
    """Load model and tokenizer with caching"""
    global model, tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            app.config['MODEL_NAME'],
            use_fast=True,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            app.config['MODEL_NAME'],
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        logger.info(f"Model {app.config['MODEL_NAME']} loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        raise

# Load model at startup
try:
    load_model()
except Exception as e:
    logger.critical(f"Failed to initialize application: {e}")
    exit(1)

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

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', styles=rephrase_styles)

@app.route('/rephrase', methods=['POST'])
def rephrase():
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 415
        
        data = request.get_json()
        original_text = data.get('sentence', '').strip()
        style_key = data.get('style', 'standard')

        if not original_text:
            return jsonify({'error': 'Sentence is required'}), 400
        if len(original_text) > 1000:
            return jsonify({'error': 'Text too long'}), 400
        if style_key not in rephrase_styles:
            style_key = 'standard'

        style_instruction = rephrase_styles[style_key]

        messages = [
            {"role": "system", "content": f"You are a helpful assistant that rephrases sentences. {style_instruction}"},
            {"role": "user", "content": f"Please rephrase this text: \"{original_text}\""}
        ]

        text = (tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if hasattr(tokenizer, 'apply_chat_template')
                else f"System: You are a helpful assistant that rephrases sentences. {style_instruction}\n"
                     f"User: Please rephrase this text: \"{original_text}\"\nAssistant: ")

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
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
        logger.info(f"Successfully rephrased text with style: {style_key}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in rephrase endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

if __name__ == '__main__':
    
    logger.info(f"Starting server on {app.config['HOST']}:{app.config['PORT']}")
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        threaded=True,
        use_reloader=False 
    )
