"""Recall lower-bound methods. Add a method: write its file, then add its class to METHODS below."""
from .binomial import BinomialBound
from .global_margin import GlobalMargin
from .scaled_margin import ScaledMargin

METHODS = {cls.name: cls for cls in (BinomialBound, GlobalMargin, ScaledMargin)}