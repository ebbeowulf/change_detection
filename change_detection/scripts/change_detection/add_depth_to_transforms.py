import json
import os
import argparse
import pdb

def add_depth_paths(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    pdb.set_trace()
    for frame in data.get('frames', []):
        color_path = frame.get('file_path')
        if color_path:
            filename = os.path.basename(color_path)
            depth_path = os.path.join('depth', filename.replace('color_', 'depth_'))
            frame['depth_file_path'] = depth_path

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Add depth_file_path to each frame in a JSON file.")
    parser.add_argument("input_json", help="Path to the input JSON file")
    parser.add_argument("output_json", help="Path to the output JSON file with depth_file_path added")
    args = parser.parse_args()

    add_depth_paths(args.input_json, args.output_json)

if __name__ == "__main__":
    main()