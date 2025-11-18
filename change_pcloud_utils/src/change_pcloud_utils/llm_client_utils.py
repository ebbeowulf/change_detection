# client.py
import socket
import json
import struct
import io
import numpy as np
from PIL import Image
import cv2

HOST = 'localhost'
PORT = 5001
BUFFER_SIZE = 4096

# Function for communicating with llm server
#    accepts both text and images by default
def send_data(text, numpy_images):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))

        header = {
            "message": text,
            "image_count": len(numpy_images)
        }
        header_bytes = json.dumps(header).encode()
        s.sendall(struct.pack('!I', len(header_bytes)))
        s.sendall(header_bytes)

        for idx, arr in enumerate(numpy_images):
            # Convert NumPy array to PNG bytes
            arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(arr_rgb.astype(np.uint8))
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            data = buffer.getvalue()

            name = f"image_{idx}.png"
            name_bytes = name.encode()

            s.sendall(struct.pack('!I', len(name_bytes)))
            s.sendall(name_bytes)
            s.sendall(struct.pack('!I', len(data)))
            s.sendall(data)

        response = s.recv(BUFFER_SIZE)
        return json.loads(response.decode())
    