from langchain_core.tools import tool
from langchain_ollama.chat_models import ChatOllama
from chat import get_response  # Assuming you have a `get_response` function elsewhere
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
import random
import torch
import base64
from PIL import Image
from flask import Flask,request,jsonify
import io

app=Flask(__name__)
# Initialize the model
llm = ChatOllama(model='llama3.1', temperature=0.9)
@tool
def create_image(prompt):
    """
    Generates a basic image based on the provided prompt using Stable Diffusion.
    """
    try:
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        pipe = pipe.to("mps")  # Ensure compatibility with your hardware
        image = pipe(prompt).images[0]

        # Convert image to base64 and return
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format='PNG')
        img_byte_array.seek(0)
        image_base64 = base64.b64encode(img_byte_array.read()).decode('utf-8')

        return image_base64
    except Exception as e:
        return str(e)

@tool
def generate_image_with_logo(prompt):
    """
    Generates a basic image based on the provided prompt logo using Stable Diffusion.
    """
    logo=Image.open('logo.png')
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
                                                        variant="fp16", use_safetensors=True).to('cuda')

        # Generate image based on the provided prompt
        generator = torch.manual_seed(random.randint(1232323,1489341482))

        image = pipe(
            prompt='generate poster with texts' + prompt,
            negative_prompt="ugly, deformed, noisy, blurry,low contrast, watermark",
            num_inference_steps=50,
            generator=generator,
            guidance_scale=15.0,
            denoise=1.0
        ).images[0]

        logo_width = int(image.size[0] * 0.15)  # 15% of canvas width
        logo_height = int(logo.height * (logo_width / logo.width))  # Maintain aspect ratio
        logo = logo.resize((logo_width, logo_height), resample=Image.Resampling.LANCZOS)
        logo = logo.convert("RGBA")

        # Randomly place the logo in the left or right corner
        logo_position = (10, 10) if random.choice([True, False]) else (image.size[0] - logo_width - 10, 10)

        # Paste the logo onto the canvas
        image.paste(logo, logo_position, logo)

        # Convert image to base64 and return
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format='PNG')
        img_byte_array.seek(0)
        image_base64 = base64.b64encode(img_byte_array.read()).decode('utf-8')

        return image_base64
    except Exception as e:
        return str(e)

# Tool to handle generic questions like "how are you?" using LLM
@tool
def respond_to_question(question, history):
    """This function is responsible for responding to general questions for the user"""
    response = get_response(user_input=question, conversation_history=history)
    print(response)
    return response.content

# Bind tools together
tools = [create_image,generate_image_with_logo,respond_to_question]
llm_with_tools = llm.bind_tools(tools=tools)

def call_method(llm_respond):
    """
    Calls a function dynamically based on the 'name' key in the provided dictionary.
    """
    # Ensure there's at least one tool call in the response
    if len(llm_respond.tool_calls) > 0:
        data = llm_respond.tool_calls[0]  # Extract the first tool call data
        print(f"Tool Call Data: {data}")

        # Extract function name and arguments from the dictionary
        func_name = data.get('name')
        args = data.get('args', {})

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


@app.route('/process',methods=["POST"])
def process():
    if 'prompt' not in request.form:
        return jsonify({"error": "No prompt provided"}), 400
    # Get the file and the prompt
    prompt = request.form['prompt']
    res=llm_with_tools.invoke(prompt)
    res=call_method(res)
    print(res)
    return res
    
    
if __name__=='__main__':
    app.run()