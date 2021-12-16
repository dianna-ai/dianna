import numpy as np
from skimage.transform import resize
from tqdm import tqdm
from dianna.utils import get_function, to_xarray, move_axis


def normalize(saliency, n_masks, p_keep):
    return saliency / n_masks / p_keep


class RISE:
    """
    RISE implementation based on https://github.com/eclique/RISE/blob/master/Easy_start.ipynb
    """

    def __init__(self, n_masks=1000, feature_res=8, p_keep=0.5,  # pylint: disable=too-many-arguments
                 axes_labels=None, preprocess_function=None):
        """RISE initializer.

        Args:
            n_masks (int): Number of masks to generate.
            feature_res (int): Resolution of features in masks.
            p_keep (float): Fraction of image to keep in each mask
            axes_labels (dict/list): If a dict, key,value pairs of axis index, name.
                                     If a list, the name of each axis where the index in the list is the axis index
            preprocess_function (callable, optional): Function to preprocess input data with
        """
        self.n_masks = n_masks
        self.feature_res = feature_res
        self.p_keep = p_keep
        self.preprocess_function = preprocess_function
        self.masks = None
        self.predictions = None
        self.axes_labels = axes_labels if axes_labels is not None else []
        self.required_labels = ('batch', 'channels')

    def explain_text(self, model_or_function, input_text, labels=(0,), batch_size=100):
        runner = get_function(model_or_function, preprocess_function=self.preprocess_function)
        input_tokens = np.asarray(model_or_function.tokenizer(input_text))
        text_length = len(input_tokens)
        self.masks = self._generate_masks_for_text(text_length)  # Expose masks for to make user inspection possible
        sentences = self._create_masked_sentences(input_tokens)
        saliencies = self._get_saliencies(runner, sentences, text_length, batch_size)
        return self._reshape_result(input_tokens, labels, saliencies)

    @staticmethod
    def _reshape_result(input_tokens, labels, saliencies):
        word_lengths = [len(t) for t in input_tokens]
        word_indices = [sum(word_lengths[:i]) + i for i in range(len(input_tokens))]
        return [list(zip(input_tokens, word_indices, saliencies[label])) for label in labels]

    def _get_saliencies(self, runner, sentences, text_length, batch_size):
        self.predictions = self._get_predictions(sentences, runner, batch_size)
        unnormalized_saliency = self.predictions.T.dot(self.masks.reshape(self.n_masks, -1)).reshape(-1, text_length)
        return normalize(unnormalized_saliency, self.n_masks, self.p_keep)

    def _get_predictions(self, sentences, runner, batch_size):
        predictions = []
        for i in tqdm(range(0, self.n_masks, batch_size), desc='Explaining'):
            predictions.append(runner(sentences[i:i + batch_size]))
        predictions = np.concatenate(predictions)
        return predictions

    def _create_masked_sentences(self, tokens):
        tokens_masked = []
        for mask in self.masks:
            tokens_masked.append(tokens[mask])
        sentences = [" ".join(t) for t in tokens_masked]
        return sentences

    def _generate_masks_for_text(self, input_size):

        masks = np.random.choice(a=(True, False), size=(self.n_masks, input_size), p=(self.p_keep, 1 - self.p_keep))
        return masks

    def explain_image(self, model_or_function, input_data, batch_size=100):
        """Run the RISE explainer.
           The model will be called with masked images,
           with a shape defined by `batch_size` and the shape of `input_data`

        Args:
            model_or_function (callable or str): The function that runs the model to be explained _or_
                                                 the path to a ONNX model on disk.
            input_data (np.ndarray): Image to be explained
            batch_size (int): Batch size to use for running the model.

        Returns:
            Explanation heatmap for each class (np.ndarray).
        """
        runner = get_function(model_or_function, preprocess_function=self.preprocess_function)
        # convert data to xarray
        input_data = to_xarray(input_data, self.axes_labels, required_labels=self.required_labels)
        # batch axis should always be first
        input_data = move_axis(input_data, 'batch', 0)
        # ensure channels axis is last and keep track of where it was so we can move it back
        channels_axis_index = input_data.dims.index('channels')
        input_data = move_axis(input_data, 'channels', -1)

        # data shape without batch axis and channel axis
        img_shape = input_data.shape[1:3]
        self.masks = self.generate_masks_for_images(img_shape)  # Expose masks for to make user inspection possible

        predictions = []

        # Make sure multiplication is being done for correct axes
        masked = (input_data * self.masks)
        # ensure channels axis is in original location again
        masked = move_axis(masked, 'channels', channels_axis_index)
        # convert to numpy for onnx
        masked = masked.values.astype(input_data.dtype)

        for i in tqdm(range(0, self.n_masks, batch_size), desc='Explaining'):
            predictions.append(runner(masked[i:i + batch_size]))
        predictions = np.concatenate(predictions)
        self.predictions = predictions
        saliency = predictions.T.dot(self.masks.reshape(self.n_masks, -1)).reshape(-1, *img_shape)
        return normalize(saliency, self.n_masks, self.p_keep)

    def generate_masks_for_images(self, input_size):
        """Generate a set of random masks to mask the input data

        Args:
            input_size (int): Size of a single sample of input data, for images without the channel axis.
        Returns:
            The generated masks (np.ndarray)
        """
        cell_size = np.ceil(np.array(input_size) / self.feature_res)
        up_size = (self.feature_res + 1) * cell_size

        grid = np.random.choice(a=(True, False), size=(self.n_masks, self.feature_res, self.feature_res),
                                p=(self.p_keep, 1 - self.p_keep))
        grid = grid.astype('float32')

        masks = np.empty((self.n_masks, *input_size))

        for i in tqdm(range(self.n_masks), desc='Generating masks'):
            y = np.random.randint(0, cell_size[0])
            x = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            masks[i, :, :] = self._upscale(grid[i], up_size)[y:y + input_size[0], x:x + input_size[1]]
        masks = masks.reshape(-1, *input_size, 1)
        return masks

    def _upscale(self, grid_i, up_size):
        return resize(grid_i, up_size, order=1, mode='reflect', anti_aliasing=False)
