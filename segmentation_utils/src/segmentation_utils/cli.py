#!/usr/bin/env python3
"""
Command-line interface for segmentation_utils.

This module provides CLI entry points for segmentation operations.
Exposes functions from the various segmentation models.
"""

import argparse
import sys
from pathlib import Path


def main():
    """
    Main entry point for the segmentation-utils-cli command.
    
    Example:
        segmentation-utils-cli --help
        segmentation-utils-cli segment --model yolo --image /path/to/image.jpg
    """
    parser = argparse.ArgumentParser(
        description="Segmentation utilities CLI",
        prog="segmentation-utils-cli"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Example subcommand: segment
    segment_parser = subparsers.add_parser("segment", help="Segment an image")
    segment_parser.add_argument(
        "--model",
        type=str,
        choices=["yolo", "dino", "clip"],
        default="yolo",
        help="Which segmentation model to use (default: yolo)"
    )
    segment_parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    segment_parser.add_argument(
        "--output",
        type=str,
        help="Path to save output (optional)"
    )
    segment_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold (default: 0.5)"
    )
    
    # Example subcommand: list-models
    list_parser = subparsers.add_parser("list-models", help="List available models")
    
    args = parser.parse_args()
    
    if args.command == "segment":
        return cmd_segment(args)
    elif args.command == "list-models":
        return cmd_list_models(args)
    else:
        parser.print_help()
        return 0


def cmd_segment(args):
    """Execute segmentation on an image."""
    image_path = Path(args.image)
    
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}", file=sys.stderr)
        return 1
    
    try:
        # Example: instantiate the appropriate model and process
        if args.model == "yolo":
            from segmentation_utils.yolo_segmentation import yolo_segmentation
            model = yolo_segmentation()
        elif args.model == "dino":
            from segmentation_utils.dino_segmentation import dino_segmentation
            model = dino_segmentation()
        elif args.model == "clip":
            from segmentation_utils.clip_segmentation import clip_segmentation
            model = clip_segmentation()
        
        # Process the file
        model.process_file(str(image_path), threshold=args.threshold, save_fileName=args.output)
        
        print(f"Segmentation completed for {image_path}")
        if args.output:
            print(f"Output saved to {args.output}")
        
        return 0
    except Exception as e:
        print(f"Error during segmentation: {e}", file=sys.stderr)
        return 1


def cmd_list_models(args):
    """List available segmentation models."""
    models = [
        "yolo - YOLO segmentation",
        "dino - DINO object detection and segmentation",
        "clip - CLIP-based segmentation"
    ]
    print("Available segmentation models:")
    for model in models:
        print(f"  - {model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
