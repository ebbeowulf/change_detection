"""
segmentation_utils - Middleware interface for HuggingFace segmentation models.

Provides unified access to multiple segmentation approaches:
- YOLO-based instance segmentation
- DINO object detection
- CLIP semantic understanding
"""

__version__ = "0.1.0"

# Optional: import main classes for convenience
try:
    from segmentation_utils.segmentation import image_segmentation
except ImportError:
    pass
