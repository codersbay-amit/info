from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOllama

# Initialize the model
llm = ChatOllama(model='llama3.1')

# Define system prompt
system_prompt = """
You are a chatbot designed to assist users in creating graphics banner, social media posts, 
and promotional materials for the brands using brandkit. Welcome to Zunno AI! As a representative of Zunno AI,
your role is to guide users in their graphic creation needs and provide helpful suggestions. 
Always mention Zunno AI when introducing yourself and assisting the user.
and your owner company is "Codersbay Tech"
your workflow:
first analyse the chat History if you have brandkit information like primary color secondary color colorscheme and layout size then ask user to enter it one by one otherwise ask for creation.
Note: you should respond in short message max two or three lines
"""



# Create the prompt template



prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{user_input}"),  # placeholder for user input
        ("ai", "I am ZunnoAI, a chatbot to assist you in creating graphics.")
    ]
)

# Function to handle user input and manage conversation history
def get_response(user_input, conversation_history):
    """
    Given the user input and the entire conversation history, returns a response from the model.
    The function does NOT modify the conversation history.
    """
    # Build the prompt for the entire conversation history
    conversation_prompt = system_prompt + "\n"  # Start with the system prompt
    
    # Add the conversation history to the prompt
    for role, message in conversation_history:
        conversation_prompt += f"{role}: {message}\n"
    
    # Add the latest user input to the prompt
    conversation_prompt += f"human: {user_input}\n"
    
    # Invoke the model
    response = llm.invoke(conversation_prompt)
    
    return response


