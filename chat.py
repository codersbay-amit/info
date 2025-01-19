from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
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
Note: you should respond in short message max two or three lines after that you need to call the image creation function
"""
system_prompt2="""Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt2),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)



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

import ollama

# Initialize Ollama client
ollama_api = ollama

def rewrite_prompt_based_on_history(prompt,chat_history ):
    """
    This function rewrites the given prompt using the chat history to make it
    clearer and more understandable with Ollama Llama 3.1 model.
    :param chat_history: List of previous conversations [(user_input, assistant_response), ...]
    :param prompt: The new prompt that needs clarification
    :return: Rewritten prompt
    """
    
    # Create a conversation context from the history
    context = ""
    for role, message in chat_history:
        context += f"{role}: {message}\n"
    
    # Add the new prompt to the context
    context += f"User: {prompt}\nAssistant:"

    # Use Ollama's Llama 3.1 to rewrite the prompt
    try:
        response = ollama_api.chat(model="llama3.1", messages=[{"role": "user", "content": f"Given the following chat history, can you rewrite the user's latest prompt to make it clearer?\n\n{context}"}])
        
        # Extract the rewritten prompt from the response
        rewritten_prompt = response["text"].strip()
        print(rewritten_prompt)
        return rewritten_prompt
    except Exception as e:
        print("Error:", e)
        return None

# Example chat history and prompt



