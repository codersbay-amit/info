import re
import json

# Original JSON string with comment
json_str = """Here's a revised version of the conversation with an added step to generate the image:

aiBased on our conversation, here's your finalized design in JSON format:

```
{
  "primary_color": "Blue",
  "secondary_color": "White",
  "background": {
    "style": "dotted",
    "color": "Blue"
  },
  "size": "Mobile poster size", // assuming standard mobile poster size
  "title": "Unlock Exclusive Deals",
  "subtitle": "",
  "action_button_text": ""
}
"""

# Remove comments from the JSON string



