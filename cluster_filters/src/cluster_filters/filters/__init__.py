"""
Filter registry: the name used in --stages -> filter class. To add a filter, write it in its own file
(inheriting from a class in base.py) and add one line here.
"""
from .siamese import SiameseFilter
from .vlm_change import VLMChangeFilter

FILTERS = {cls.name: cls for cls in (SiameseFilter,VLMChangeFilter)}