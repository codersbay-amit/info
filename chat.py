from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOllama

# Initialize the model
llm = ChatOllama(model='llama3.1')

# Define system prompt
system_prompt = """
You are a chatbot designed to assist users in creating graphics such as images, social media posts, 
and promotional materials. Welcome to ZunnoAI! As a representative of ZunnoAI,
 your role is to guide users in their graphic creation needs and provide helpful suggestions. 
 Always mention ZunnoAI when introducing yourself and assisting the user.
"""
# Define a conversation history manager


# Create a template for the chat prompt
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{user_input}"),  # placeholder for user input
        ("ai", "I am ZunnoAI, a chatbot to assist you in creating graphics.")
    ]
)

# Function to handle user input and manage conversation history
def get_response(user_input,conversation_history=[]):
    # Add user's input to the conversation history
    conversation_history.append(("human", user_input))
    
    # Create the prompt for the current conversation
    prompt = prompt_template.format_messages(user_input=user_input)

    # Invoke the model
    response = llm.invoke(prompt)
    
    # Add AI's response to the conversation history
    conversation_history.append(("ai", response))
    
    return response

# Example: Handling user input and getting a response

