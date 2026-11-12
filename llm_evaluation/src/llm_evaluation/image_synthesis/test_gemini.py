import base64
from google import genai

client = genai.Client()  # reads GEMINI_API_KEY from the environment
img = "/data3/datasets/coco/coco/train2017/000000000138.jpg"
data = base64.b64encode(open(img, "rb").read()).decode()

prompt = ("Could a forklift realistically be placed in this scene? Answer yes or no first. "
          "If yes, detect the floor or ground surface it would rest on. Return JSON: a list of "
          "objects with box_2d as [ymin, xmin, ymax, xmax] normalized to 0-1000, mask as a "
          "polygon of [x, y] points normalized to 0-1000, and label.")

interaction = client.interactions.create(
    model="gemini-3.5-flash",
    input=[{"type": "text", "text": prompt},
           {"type": "image", "data": data, "mime_type": "image/jpeg"}],
)
print(interaction.output_text)