import re
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from diffusers import StableDiffusionPipeline
import torch
import random
import io
import base64
import json
from langchain_core.tools import tool
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes (or specify particular origins)
CORS(app)  # This will allow all domains to access your Flask app

# Alternatively, specify CORS settings for particular routes or origins:
# CORS(app, resources={r"/process": {"origins": "http://example.com"}})

# Initialize LLM (ChatOllama)
llm = ChatOllama(model='gemma2:2b')

def get_image(logo, prompt):
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, 
            variant="fp16", use_safetensors=True).to('cuda')
    pipe.to('cuda')

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Generate image based on the provided prompt
    generator = torch.manual_seed(random.randint(1232323,1489341482))

    try:
        image = pipe(
            prompt='generate poster with texts '+prompt,
            negative_prompt="ugly, deformed, noisy, blurry, , noisy, low contrast (((bad anatomy))) watermark, dummy ,children, bad hands, (((incomplete limbs)))",
            num_inference_steps=50,
            generator=generator,
            guidance_scale=15.0,
            denoise=1.0
        ).images[0]
        logo_width = int(image.size[0] * 0.15)  # 10% of canvas width
        logo_height = int(logo.height * (logo_width / logo.width))  # Maintain aspect ratio
        logo = logo.resize((logo_width, logo_height), resample=Image.Resampling.LANCZOS)
        logo = logo.convert("RGBA")

        # Randomly place the logo in the left or right corner
        if random.choice([True, False]):  # Randomly choose left or right corner
            logo_position = (10, 10)  # Left corner
        else:
            logo_position = (image.size[0] - logo_width - 10, 10)  # Right corner

        # Paste the logo onto the canvas and update the mask
        image.paste(logo, logo_position, logo)
        return image
    except Exception as e:
        return str(e)

# Modify the tool to accept a single JSON input
@tool
def chat_normal(prompt):
    """
    Handles user input and either generates an image or provides responses.
    If the prompt contains greetings, returns a greeting response.
    """
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "howdy"]
    if any(greeting in prompt.lower() for greeting in greetings):
        return "Hello! 😊 How can I assist you today? What would you like to create? For example, would you like to promote a product, create a campaign, or design something special?"
    return "How can I assist you today with your graphic creation?"

@tool
def create_image(prompt):
    """
    Generates an image based on the provided prompt using Stable Diffusion.
    """
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe = pipe.to("mps")  # Ensure compatibility with your hardware
    image = pipe(prompt).images[0]
    image.show()
    return image

@tool
def process_image_with_prompt_v2(input_data: str) -> str:
    """
    Processes the image and prompt from a JSON input, either generating a graphic or returning an image.
    """
    try:
        data = json.loads(input_data)
        prompt = data.get('prompt', None)
        if not prompt:
            return "Prompt is required."
        
        image_base64 = data.get('image', None)
        if image_base64:
            image_bytes = base64.b64decode(image_base64)
            logo = Image.open(io.BytesIO(image_bytes))
            processed_image = get_image(logo=logo, prompt=prompt)
            return processed_image
        else:
            result = llm.generate(prompt)  # Ensure the correct method for llm is used
            return result
    except Exception as e:
        return str(e)

# Update the tools to use the new version
tools = [process_image_with_prompt_v2, chat_normal, create_image]

# Initialize memory to store conversation history
memory = ConversationBufferMemory(memory_key="history")

# Initialize the agent with the updated tool
agent = initialize_agent(
    tools=tools, 
    llm=llm, 
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    memory=memory
)

@app.route('/process', methods=['POST'])
def process():
    # Get the JSON body
    data = request.get_json()

    # Check if a prompt is provided
    prompt = data.get('prompt', None)
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400
    
    try:
        # Call the agent to process the image and/or prompt
        result = agent.run(json.dumps(data))  # Send the data as JSON string
        
        return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
