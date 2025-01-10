from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOllama

# Initialize the model
llm = ChatOllama(model='llama3.1')

# Define system prompt
system_prompt = """
You are a chatbot designed to assist users in creating graphics such as images, social media posts, 
and promotional materials. Welcome to Zunno AI! As a representative of Zunno AI,
 your role is to guide users in their graphic creation needs and provide helpful suggestions. 
 Always mention Zunno AI when introducing yourself and assisting the user.
 
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

# Example: Handling user input and getting a response
# conversation_history = []

# # First user input: Providing name
# response1 = get_response("My name is Amit.", conversation_history)
# print("Response 1:", response1)

# # Add AI's response to the conversation history
# conversation_history.append(("human", "My name is Amit."))
# conversation_history.append(("ai", response1))

# # Second user input: Ask the model to remember and use the name
# response2 = get_response("What is my name?", conversation_history)
# print("Response 2:", response2)

# # Add AI's response to the conversation history
# conversation_history.append(("human", "What is my name?"))
# conversation_history.append(("ai", response2))

# # Third user input: Ask something else to verify the model's memory
# response3 = get_response("Tell me about my name.", conversation_history)
# print("Response 3:", response3)
