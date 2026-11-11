"""The two onnx.helper functions dianna uses."""
import numpy as np
from onnx.onnx_ml_pb2 import AttributeProto
from onnx.onnx_ml_pb2 import NodeProto
from onnx.onnx_ml_pb2 import TensorProto


def make_node(op_type, inputs, outputs, **attributes):
    """Like onnx.helper.make_node, for integer attributes only."""
    node = NodeProto(op_type=op_type, input=inputs, output=outputs)
    for name, value in attributes.items():
        node.attribute.add(name=name, i=value, type=AttributeProto.INT)
    return node


def tensor_dtype_to_np_dtype(tensor_dtype):
    """Numpy dtype for an ONNX TensorProto element type."""
    return np.dtype({
        TensorProto.FLOAT: 'float32',
        TensorProto.DOUBLE: 'float64',
        TensorProto.INT32: 'int32',
        TensorProto.INT64: 'int64',
        TensorProto.UINT8: 'uint8',
        TensorProto.BOOL: 'bool'
    }[tensor_dtype])
