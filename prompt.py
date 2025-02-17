from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder,PromptTemplate
from langchain.chat_models import ChatOllama

# Initialize the model
llm = ChatOllama(model='llama3.1',temprature=0.95)

# Define system prompt


# Function to handle user input and manage conversation history
def geneate_prompt(data,user_input="genearate prompt",):
    data=f"""{data}"""

    prompt = f"""


    generate prompt according to the history and image should have the only one image of product (if specified) and prompt should be short (70 word maximum) and use this data
    {data}



    note: your prompt should be like this (use this pattern to generate prompt for image generation) :
        ' an advertisement for , ("letest_user_quesry" user context), simple design, with the title ("title_text" text logo),with the subtitle ("subtitle_text" text logo),with the button_text ("button_text" text logo), clean design, vibrant images, product photography, blank dark_blue background '


    and prompt should short and consize

    return only prompt not the explaination

    """

  

    response=llm.invoke(prompt).content+"<lora:Harrlogos_v1.1:1>"
    print(response)
    return response





data={
    "product_name":"shoe",
    "primary_color":"red",
    "secondary_color":"white",
    "background":"plain white",
    "size":"1024x1024",
    "title":"black friday sale",
    "subtitle":"amazing offer",
    "action_button_text":"shop now"
}

geneate_prompt(data=data)