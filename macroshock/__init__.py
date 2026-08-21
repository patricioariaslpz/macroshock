# __init__.py
# Package entry point: re-exports the SVAR class so users can write
# `from macroshock import SVAR` instead of `from macroshock.macroshock import SVAR`.
from .macroshock import SVAR