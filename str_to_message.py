from langchain import HumanMessage, AIMessage

def convert_conversation_to_messages(conversation: str):
    """
    Convert a conversation string into an array of HumanMessage and AIMessage objects.

    Args:
    - conversation (str): A conversation with 'user:' and 'bot:' as identifiers.

    Returns:
    - list: A list of HumanMessage and AIMessage objects.
    """
    # Split the conversation into lines
    lines = conversation.split('\n')

    # Initialize an empty list to store the messages
    messages = []

    # Iterate over the conversation lines
    for line in lines:
        if line.startswith('bot:'):
            # Create AIMessage if the line starts with 'bot:'
            ai_message = AIMessage(content=line.replace('bot: ', ''))
            messages.append(ai_message)
        elif line.startswith('user:'):
            # Create HumanMessage if the line starts with 'user:'
            human_message = HumanMessage(content=line.replace('user: ', ''))
            messages.append(human_message)

    return messages


