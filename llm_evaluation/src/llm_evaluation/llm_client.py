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


# 🔧 Test mode: run from terminal
if __name__ == '__main__':
    import sys
    import os

    if len(sys.argv) < 3:
        print("Usage: python client.py 'your message here' image1 [image2 ... image4]")
        sys.exit(1)

    message = sys.argv[1]
    image_files = sys.argv[2:]

    if not (1 <= len(image_files) <= 4):
        print("Please provide between 1 and 4 image files.")
        sys.exit(1)

    image_list=[]
    for img in image_files:
        if not os.path.isfile(img):
            print(f"File not found: {img}")
            sys.exit(1)
        image_list.append(cv2.imread(img,-1))

    result = send_data(message, image_list)
    print("Server response:", json.dumps(result, indent=2))