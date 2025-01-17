from langchain_core.tools import tool
from langchain_ollama.chat_models import ChatOllama
from chat import get_response  # Assuming you have a `get_response` function elsewhere
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
import random
import torch
import base64
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
from langchain_core.messages import HumanMessage,AIMessage
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Initialize the model
llm = ChatOllama(model='llama3.1', temperature=0.9)

session_history = {}

@tool
def generate_image_with_logo(prompt,session_id):
    """
    Generates a basic image based on the provided prompt with a logo using Stable Diffusion.
    and prompt should have brandkit information then this function will not work
    """
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
                                                        variant="fp16", use_safetensors=True).to('cuda')
        # Generate image based on the provided prompt
        generator = torch.manual_seed(random.randint(1232323, 1489341482))
        image = pipe(
            prompt='generate a poster with visual texts ' + prompt,
            negative_prompt="ugly, deformed, noisy, blurry,low contrast, watermark",
            num_inference_steps=50,
            generator=generator,
            guidance_scale=15.0,
            denoise=1.0
        ).images[0]
        # logo_width = int(image.size[0] * 0.15)  # 15% of canvas width
        # logo_height = int(logo.height * (logo_width / logo.width))  # Maintain aspect ratio
        # logo = logo.resize((logo_width, logo_height), resample=Image.Resampling.LANCZOS)
        # logo = logo.convert("RGBA")
        # # Randomly place the logo in the left or right corner
        # logo_position = (10, 10) if random.choice([True, False]) else (image.size[0] - logo_width - 10, 10)
        # # Paste the logo onto the canvas
        # image.paste(logo, logo_position, logo)

        # Convert image to base64 and return
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format='PNG')
        img_byte_array.seek(0)
        image_base64 = base64.b64encode(img_byte_array.read()).decode('utf-8')

        return jsonify({"type": "base64", "data": image_base64})
    except Exception as e:
        return str(e)


@tool
def respond_to_question(question, session_id):
    """This function is responsible for responding to general questions for the user"""
    # Update the conversation history for this session
    if session_id not in session_history:
        session_history[session_id] = []

    # Append user query to history
    session_history[session_id].append(("human", question))
    # Get response using external method (this could be a language model, etc.)
    response = get_response(user_input=question, conversation_history=session_history[session_id])

    # Append the assistant's response to the history
    session_history[session_id].append(("ai", response.content))
    
    return jsonify({"type": "text", "data": response.content})


# Bind tools together
tools = [ generate_image_with_logo, respond_to_question]
llm_with_tools = llm.bind_tools(tools=tools)


def call_method(llm_respond, session_id):
    """
    Calls a function dynamically based on the 'name' key in the provided dictionary.
    """
    # Ensure there's at least one tool call in the response
    if len(llm_respond.tool_calls) > 0:
        data = llm_respond.tool_calls[0]  # Extract the first tool call data
 
        # Extract function name and arguments from the dictionary
        func_name = data.get('name')
        args = data.get('args', {})

        # Include session_id in the args for history management
        args['session_id'] = session_id

        # Check if the function exists in the global namespace
        func = globals().get(func_name)

        # If the function is found, call it with the arguments
        if func:
            print(f"Calling function {func_name} with arguments: {args}")
            return func.invoke(args)
        else:
            return f"Function '{func_name}' not found."
    else:
        return "No tool calls found in the response."


@app.route('/process', methods=["POST"])
def process():
    if 'prompt' not in request.form or 'session_id' not in request.form:
        return jsonify({"error": "No prompt or session_id provided"}), 400
    
   

    # Get the session_id and prompt from the request
    prompt = request.form['prompt']
    session_id = request.form['session_id']

    if 'logo' in request.files:
        logo=request.files['logo']
        logo.save("logo"+session_id+".png")
        prompt=prompt+" with this logo"

    # Process the prompt with the llm

    res = llm_with_tools.invoke(prompt)
    
    # Call the appropriate method for the tool used
    res = call_method(res, session_id)
  
    return res


if __name__ == '__main__':
    app.run(debug=True)