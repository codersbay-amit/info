from langchain_core.tools import tool
from langchain_ollama.chat_models import ChatOllama
from prompt import geneate_prompt
from chat import get_response  # Assuming you have a `get_response` function elsewhere
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
import random
from refiner import refiner
import torch
from peft import PeftModel
import base64
# from sd_3_5 import generate
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from test import check_json_in_string
from image_with_product import generate_image_with_product
import io
from langchain_core.messages import HumanMessage,AIMessage
import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Initialize the model
llm = ChatOllama(model='llama3.1', temperature=0.9)

session_history = {}


def generate_image(prompt):
        """
        Generates a basic image based on the provided prompt with a logo using Stable Diffusion.
        and prompt should have brandkit information then this function will not work
        """
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
                                                        variant="fp16", use_safetensors=True).to('cuda')
       
        generator = torch.manual_seed(random.randint(1232323, 1489341482))
        image = pipe(
            prompt='realistic ' + prompt,
            negative_prompt="ugly, deformed, noisy, blurry,low contrast, watermark",
            num_inference_steps=50,
            generator=generator,
            guidance_scale=15.0,
            denoise=1.0
        ).images[0]
        
        return image
def generate_image_with_logo(prompt):
        """
        Generates a basic image based on the provided prompt with a logo using Stable Diffusion.
        and prompt should have brandkit information then this function will not work
        """
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
                                                        variant="fp16", use_safetensors=True).to('cuda')
       
        generator = torch.manual_seed(random.randint(1232323, 1489341482))
        image = pipe(
            prompt=' text aided realistic, promtional poster with logo ' + prompt,
            negative_prompt="ugly, deformed, noisy, blurry,low contrast, watermark,miss spelled",
            num_inference_steps=50,
            generator=generator,
            guidance_scale=15.0,
            denoise=1.0
        ).images[0]
        
        return image
        # # Convert image to base64 and return
        # img_byte_array = io.BytesIO()
        # image.save(img_byte_array, format='PNG')
        # img_byte_array.seek(0)
        # image_base64 = base64.b64encode(img_byte_array.read()).decode('utf-8')

        # return jsonify({"type": "base64", "data": image_base64})
  

@tool
def respond_to_question(question, session_id):
    """This function is responsible for responding to general questions for the user"""
    # Update the conversation history for this session
    if session_id not in session_history:
        session_history[session_id] = []

    # Append user query to history
    session_history[session_id].append("human:"+str(question)+"\n")
    # Get response using external method (this could be a language model, etc.)
    response = get_response(user_input=question, conversation_history=session_history[session_id])

    data=check_json_in_string(response.content)
    session_history[session_id].append("ai"+response.content+"\n")
    print(response.content)
    if data is not None:

        if "prompt" in data.keys():
            if "product_url" in data.keys():
                return generate_image_with_product(prompt=data['prompt'],image_url=data['product_url'])
            else:
                print("using prompt")
                image= generate_image(data['prompt'])
                img_byte_array = io.BytesIO()
                image.save(img_byte_array, format='PNG')
                img_byte_array.seek(0)
                image_base64 = base64.b64encode(img_byte_array.read()).decode('utf-8')
                return jsonify({"type": "base64", "data": image_base64})
        else:
            print("using brandkit")
            prompt=geneate_prompt(data)
            image= generate_image_with_logo(prompt)
            image.save("images/image_"+str(session_id)+".png")
            image=refiner(image_path="images/image_"+str(session_id)+".png",title=data['title'],subtitle=data['subtitle'])
            img_byte_array = io.BytesIO()
            image.save(img_byte_array, format='PNG')
            img_byte_array.seek(0)
            image_base64 = base64.b64encode(img_byte_array.read()).decode('utf-8')
            return jsonify({"type": "base64", "data": image_base64})
    else:
        # Append the assistant's response to the history
        return jsonify({"type": "text", "data": response.content})
    


# Bind tools together
tools = [ respond_to_question]
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
    res = llm_with_tools.invoke(prompt)
    
    # Call the appropriate method for the tool used
    res = call_method(res, session_id)
  
    return res


if __name__ == '__main__':
    app.run(debug=True)