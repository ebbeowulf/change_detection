import json

class visual_language_model():
    def __init__(self):
        pass

    def process_input(self, text_message, images):
        # Simulate processing the input with a visual language model
        response = {
            "status": False,
            "full_text": "not implemented",
            "json": {}
        }
        return response

    def extract_json(self, response_message):
        try:
            start_index = response_message.find('{')
            end_index = response_message.rfind('}') + 1
            json_str = response_message[start_index:end_index].replace("'", '"')
            json_out = json.loads(json_str)
            return json_out
        except Exception as e:
            return {}

