import onnxruntime as ort


class SimpleModelRunner:
    """Runs an onnx model with a set of inputs and outputs."""

    def __init__(self, filename, preprocess_function=None):
        """Generates function to run ONNX model with one set of inputs and outputs.

        Args:
            filename (str): Path to ONNX model on disk
            preprocess_function (callable, optional): Function to preprocess input data with.
                For models with multiple inputs, it should return a list/tuple with one array
                per model input, in the order of the model inputs.

        Returns:
            function

        Examples:
            >>> runner = SimpleModelRunner('path_to_model.onnx')
            >>> predictions = runner(input_data)
        """
        self.filename = filename
        self.preprocess_function = preprocess_function

    def __call__(self, input_data):
        """Get ONNX predictions."""
        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False  # disables pre-allocation of memory
        sess = ort.InferenceSession(self.filename, sess_options=sess_options)
        input_names = [node.name for node in sess.get_inputs()]
        output_name = sess.get_outputs()[0].name

        if self.preprocess_function is not None:
            input_data = self.preprocess_function(input_data)

        if not isinstance(input_data, (list, tuple)):
            input_data = [input_data]
        onnx_input = dict(zip(input_names, input_data, strict=True))
        pred_onnx = sess.run([output_name], onnx_input)[0]
        return pred_onnx
