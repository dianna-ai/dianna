"""Browser stand-in for onnxruntime (which has no Pyodide build): runs the model in onnxruntime-web.

The explainers call the model synchronously from inside their loops, so the JS promises are awaited with
JSPI (`run_sync`). Only the part of the API that dianna uses is implemented.
"""
import js
import numpy as np
from pyodide.code import run_js
from pyodide.ffi import run_sync
from pyodide.ffi import to_js

ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.mjs'
ort = run_sync(run_js(f'import("{ORT_URL}")'))
ort.env.wasm.numThreads = 1  # threads need cross-origin isolation, which a static host does not give


class SessionOptions:
    """Accepted and ignored."""


class _Node:

    def __init__(self, name):
        self.name = name


class InferenceSession:
    """An onnxruntime-web session for a model given as bytes or as a file path."""

    def __init__(self, model, sess_options=None, providers=None):
        """Create the session; the options are ignored."""
        if not isinstance(model, bytes):
            with open(model, 'rb') as f:
                model = f.read()
        self._session = run_sync(ort.InferenceSession.create(to_js(model)))

    def get_inputs(self):
        """Input nodes (only their names)."""
        return [_Node(name) for name in self._session.inputNames]

    def get_outputs(self):
        """Output nodes (only their names)."""
        return [_Node(name) for name in self._session.outputNames]

    def run(self, output_names, input_feed, run_options=None):
        """Run the model; returns the outputs as float32 arrays."""
        feeds = js.Object.new()
        for name, value in input_feed.items():
            value = np.asarray(value)
            # ponytail: picks the dtype from the data, not the model input; fine for float32 and int64 models
            dtype, array = (
                'int64', js.BigInt64Array) if value.dtype.kind in 'iu' else (
                    'float32', js.Float32Array)
            value = np.ascontiguousarray(value, dtype=dtype)
            data = array.new(to_js(value.tobytes()).buffer)
            setattr(feeds, name,
                    ort.Tensor.new(dtype, data, to_js(list(value.shape))))
        out = run_sync(self._session.run(feeds)).as_object_map()
        names = output_names or list(self._session.outputNames)
        return [
            np.asarray(out[n].data.to_py(),
                       dtype=np.float32).reshape(out[n].dims.to_py())
            for n in names
        ]
