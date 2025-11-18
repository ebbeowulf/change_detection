# server.py
import socket
import json
import struct
from vlm import visual_language_model
import pdb
from PIL import Image
import io

HOST = 'localhost'
PORT = 5001
BUFFER_SIZE = 4096


def query_llm(VLM, text_message, pil_images:list):
    result=VLM.process_input(text_message, pil_images)
    return result

def receive_all(sock, length):
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise EOFError("Socket closed")
        data += more
    return data

def handle_client(conn, VLM):
    header_len = struct.unpack('!I', conn.recv(4))[0]
    header = json.loads(receive_all(conn, header_len).decode())

    text = header['message']
    image_count = header['image_count']
    images = []

    for _ in range(image_count):
        name_len = struct.unpack('!I', conn.recv(4))[0]
        _ = receive_all(conn, name_len)  # discard name
        size = struct.unpack('!I', conn.recv(4))[0]
        data = receive_all(conn, size)

        # Decode image bytes into a PIL Image object
        image = Image.open(io.BytesIO(data)).convert('RGB')
        images.append(image)

    result = query_llm(VLM, text, images)  # Pass PIL images directly
    conn.sendall(json.dumps(result).encode())
    conn.close()

# 🔧 Test mode: run from terminal
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default=HOST, help='Host to bind the server')
    parser.add_argument('--port', type=int, default=PORT, help='Port to bind the server')
    parser.add_argument('--llm_library', type=str, default='ollama', help='LLM library to use (e.g., ollama)')
    parser.add_argument('--llm_model', type=str, default='llama4:scout', help='LLM library to use (e.g., ollama)')
    args = parser.parse_args()

    if args.llm_library == 'ollama':
        from ollama_lib import ollama_lib
        VLM = ollama_lib(args.llm_model)
    else:
        raise Exception(f"LLM library {args.llm_library} not supported")
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((args.host, args.port))
        s.listen()
        print(f"Server listening on {args.host}:{args.port}")
        while True:
            conn, _ = s.accept()
            handle_client(conn, VLM)
    print("Server shut down.")