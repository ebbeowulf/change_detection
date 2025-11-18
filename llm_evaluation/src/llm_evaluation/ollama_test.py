import os
import sys
import ollama

#PROMPT = "Describe the item in the red box in the provided image"
#PROMPT = "Identify the most likely object surrounded by the red box in the provided image and at least 2 alternative descriptions for the same object. Return only a single JSON object of format: {'objects': [{'name': <string>,'confidence': <float>}]} where confidence is between [0,1]"
PROMPT = "We are deciding on tasks for a robot that can pick stuff up and put it away. The owner of the house has told the robot not to pick up stuff that is currently being read or worked on. The object surrounded by the red box in the provided image has been identified by the robot as a candidate for cleaning. Is it an object that the robot should pick up and put away? Return an answer in JSON format as {'object': <type of object>, 'pickup': <yes/no>}"


def run_inference(model: str, image_path: str):
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT, "images": [image_path]}],
        stream=True,
    )

    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)


def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <model_name> <image_path>")
        sys.exit(1)

    model_name = sys.argv[1]
    image_path = sys.argv[2]

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        sys.exit(1)

    valid_models = ["llava:13b", "llava-llama3", "llama4:scout", "moondream"]
    if model_name not in valid_models:
        print(f"Error: Invalid model name. Choose from: {', '.join(valid_models)}")
        sys.exit(1)
    run_inference(
        sys.argv[1],
        sys.argv[2],
    )


if __name__ == "__main__":
    main()

