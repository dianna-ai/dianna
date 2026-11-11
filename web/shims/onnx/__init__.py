"""Browser stand-in for onnx (whose C++ extension has no Pyodide build): its protobuf classes and `load`.

onnx_ml_pb2.py is copied from the real onnx package by build.py.
"""
from onnx import helper  # noqa: F401
from onnx.onnx_ml_pb2 import *  # noqa: F403


def load(f):
    """Load a ModelProto from a path or a file-like object."""
    if hasattr(f, 'read'):
        return ModelProto.FromString(f.read())  # noqa: F405
    with open(f, 'rb') as file:
        return ModelProto.FromString(file.read())  # noqa: F405
