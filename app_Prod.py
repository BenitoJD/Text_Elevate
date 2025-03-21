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
    MODEL_NAME = os.getenv('MODEL_NAME', 'Qwen/Qwen2.5-3B-Instruct')
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5019))
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max request size
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
    "fix-grammar": "Correct grammar, punctuation, and spelling errors while keeping the original style and meaning intact.",
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
    "update-email": "Transform the text into a succinct, informative update with a professional structure, ideal for status reports or team notifications.",
    "simplify": "Break down complex ideas into simpler, easy-to-understand language, ideal for explaining technical concepts to non-experts.",
    "debugging-tone": "Rephrase with a problem-solving focus, using a calm and methodical tone suitable for troubleshooting or explaining fixes.",
    "code-comment": "Turn the text into a clear, concise, and descriptive style like a code comment for developers.",
    "it-support-ticket": "Format the text as a clear, detailed, and professional IT support ticket description, including key details for quick resolution.",
    "release-notes": "Rephrase the text into a structured, user-friendly format suitable for software release notes, highlighting features and fixes.",
    "escalation-tone": "Adjust the text to sound urgent yet professional, suitable for escalating an issue to higher support tiers or management.",
    "training-guide": "Revise the text into a step-by-step, instructional tone for creating user guides or developer training materials.",
    "faq-style": "Rephrase the text into a question-and-answer format, concise and clear, for IT support FAQs or knowledge bases.",
    "dev-handover": "Structure the text as a detailed, technical handover note for developers, focusing on key implementation details and next steps.",
    "meeting-notes": "Condense the text into a clear, organized summary suitable for sharing after a technical meeting or sprint review.",
    "incident-report": "Rephrase the text into a formal, factual, and detailed report style for documenting IT incidents or outages.",
    "user-story": "Format the text as a concise Agile user story (e.g., 'As a [user], I want [goal] so that [benefit]') for development teams.",
    "api-documentation": "Revise the text into a precise, structured format suitable for API docs, with technical details and examples.",
    "change-request": "Shape the text into a formal, detailed request for system or code changes, including justification and impact.",
    "root-cause-analysis": "Rephrase the text into an analytical, step-by-step breakdown for explaining the cause of an issue, aimed at technical teams.",
    "status-update": "Turn the text into a brief, factual update on project or task progress, suitable for team syncs or dashboards.",
    "rejection-email": "Craft the text into a polite, professional response declining a request or bug report, with clear reasoning.",
    "onboarding-email": "Revise the text into a welcoming, informative message for new team members or users, with key details and next steps.",
    "deprecation-notice": "Adjust the text into a clear, technical announcement about phasing out a feature or tool, with alternatives provided.",
    "post-mortem": "Structure the text as a detailed, reflective summary of an event or failure, including lessons learned, for IT/dev teams.",
    "customer-facing": "Rephrase the text to be polite, clear, and non-technical, ideal for communicating with end-users or clients.",
    "sprint-goal": "Condense the text into a focused, motivating statement for a development sprint objective.",
    "test-case": "Format the text as a structured test case with steps, expected results, and conditions, for QA or devs.",
    "log-entry": "Rephrase the text into a brief, timestamp-ready format suitable for system or application logs.",
    "motivational": "Revise the text to inspire and energize a team, with an upbeat tone for morale boosts.",
    "knowledge-article": "Transform the text into a structured, informative knowledge base article with a clear problem, solution, and additional notes for IT support or users.",
    "issue-fix-article": "Rephrase the text into a concise, step-by-step article focused on resolving a specific issue, including prerequisites and verification steps, for support teams.",
    "create-documentation": "Revise the text into a detailed, well-organized documentation format with sections like overview, instructions, and troubleshooting, suitable for IT or dev reference.",
    "free-style": "Adapt the text creatively based on context, with no strict guidelines.",
    "sql-precision": "Process the user’s SQL query with exactness—correct errors, craft new sql code, enhance performance, and explain as needed, adapting to context with a professional, focused edge."
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
                max_new_tokens=1000,
                do_sample=False,
                temperature=0.85,
                top_p=0.5,
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
