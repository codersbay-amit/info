import json
import re
from langchain_ollama import OllamaLLM



import re
import json

def check_json_in_string(input_str):
    # Regex pattern to match JSON-like sections, considering nested structures
    json_pattern = r'({[^{}]*//.*?[^{}]*|[^{}]*\n)*})'

    json_str_clean = re.sub(r'//.*', '', input_str)
    # Try to extract the JSON-like structure from the input string using regex
    json_pattern = re.search(r'(\{.*\})', json_str_clean, re.DOTALL)
    
    if json_pattern:
        json_str = json_pattern.group(1)  # Extract the matched JSON string
        
        try:
            # Attempt to parse the extracted string as JSON
            parsed_data = json.loads(json_str)
            
            # Define the required keys to check
            required_keys = {
                "product_name",
                "primary_color",
                "secondary_color",
                "background",
                "size",
                "title",
                "subtitle",
                "action_button_text"
            }
            required_keys1 = {
                "prompt"
            }
            if isinstance(parsed_data, dict) and required_keys1.issubset(parsed_data.keys()):
                return parsed_data  # Return the JSON as a dictionary

            # Check if the parsed data is a dictionary and contains the required keys
            if isinstance(parsed_data, dict) and required_keys.issubset(parsed_data.keys()):
                return parsed_data  # Return the JSON as a dictionary
            else:
                return None  # Return None if the structure is incorrect
        except json.JSONDecodeError:
            return None  # Return None if the string is not valid JSON
    else:
        return None  # Return None if no JSON structure is found



