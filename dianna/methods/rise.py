import numpy as np
from skimage.transform import resize
from tqdm import tqdm
from dianna import utils


def normalize(saliency, n_masks, p_keep):
    return saliency / n_masks / p_keep


def _upscale(grid_i, up_size):
    return resize(grid_i, up_size, order=1, mode='reflect', anti_aliasing=False)


class RISE:
    """
    RISE implementation based on https://github.com/eclique/RISE/blob/master/Easy_start.ipynb
    """
    # axis labels required to be present in input image data
    required_labels = ('batch', 'channels')

    def __init__(self, n_masks=1000, feature_res=8, p_keep=0.5,  # pylint: disable=too-many-arguments
                 axes_labels=None, preprocess_function=None):

        """RISE initializer.

        Args:
            n_masks (int): Number of masks to generate.
            feature_res (int): Resolution of features in masks.
            p_keep (float): Fraction of image to keep in each mask
            axes_labels (dict/list, optional): If a dict, key,value pairs of axis index, name.
                                               If a list, the name of each axis where the index
                                               in the list is the axis index
            preprocess_function (callable, optional): Function to preprocess input data with
        """
        self.n_masks = n_masks
        self.feature_res = feature_res
        self.p_keep = p_keep
        self.preprocess_function = preprocess_function
        self.masks = None
        self.predictions = None
        self.axes_labels = axes_labels if axes_labels is not None else []

    def explain_text(self, model_or_function, input_text, labels=(0,), batch_size=100):
        runner = utils.get_function(model_or_function, preprocess_function=self.preprocess_function)
        input_tokens = np.asarray(model_or_function.tokenizer(input_text))
        text_length = len(input_tokens)
        # p_keep = self._determine_p_keep_for_text(input_tokens, runner) if self.p_keep == None else self.p_keep
        p_keep = 0.5
        input_shape = (text_length,)
        self.masks = self._generate_masks_for_text(input_shape, p_keep,
                                                   self.n_masks)  # Expose masks for to make user inspection possible
        sentences = self._create_masked_sentences(input_tokens, self.masks)
        saliencies = self._get_saliencies(runner, sentences, text_length, batch_size, p_keep)
        return self._reshape_result(input_tokens, labels, saliencies)

    def _determine_p_keep_for_text(self, input_data, runner, n_masks=100):
        p_keeps = np.arange(0.1, 1.0, 0.1)
        stds = []
        for p_keep in p_keeps:
            std = self._calculate_mean_class_std_for_text(p_keep, runner, input_data, n_masks=n_masks)
            stds += [std]
        best_i = np.argmax(stds)
        best_p_keep = p_keeps[best_i]
        print(f'Rise parameter p_keep was automatically determined at {best_p_keep}')
        return best_p_keep

    def _calculate_mean_class_std_for_text(self, p_keep, runner, input_data, n_masks):
        batch_size = 50
        masks = self._generate_masks_for_text(input_data.shape, p_keep, n_masks)
        masked = self._create_masked_sentences(input_data, masks)
        predictions = []
        for i in range(0, n_masks, batch_size):
            current_input = masked[i:i + batch_size]
            current_predictions = runner(current_input)
            predictions.append(current_predictions)
        predictions = np.concatenate(predictions)
        std_per_class = predictions.std(axis=0)
        return np.mean(std_per_class)

    def _generate_masks_for_text(self, input_shape, p_keep, n_masks):
        masks = np.random.choice(a=(True, False), size=(n_masks,) + input_shape, p=(p_keep, 1 - p_keep))
        return masks

    def _get_saliencies(self, runner, sentences, text_length, batch_size, p_keep):  # pylint: disable=too-many-arguments
        self.predictions = self._get_predictions(sentences, runner, batch_size)
        unnormalized_saliency = self.predictions.T.dot(self.masks.reshape(self.n_masks, -1)).reshape(-1, text_length)
        return normalize(unnormalized_saliency, self.n_masks, p_keep)

    @staticmethod
    def _reshape_result(input_tokens, labels, saliencies):
        word_lengths = [len(t) for t in input_tokens]
        word_indices = [sum(word_lengths[:i]) + i for i in range(len(input_tokens))]
        return [list(zip(input_tokens, word_indices, saliencies[label])) for label in labels]

    def _get_predictions(self, sentences, runner, batch_size):
        predictions = []
        for i in tqdm(range(0, self.n_masks, batch_size), desc='Explaining'):
            predictions.append(runner(sentences[i:i + batch_size]))
        predictions = np.concatenate(predictions)
        return predictions

    def _create_masked_sentences(self, tokens, masks):
        tokens_masked = []
        for mask in masks:
            tokens_masked.append(tokens[mask])
        sentences = [" ".join(t) for t in tokens_masked]
        return sentences

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

        runner = utils.get_function(model_or_function, preprocess_function=self.preprocess_function)
        # convert data to xarray
        input_data = utils.to_xarray(input_data, self.axes_labels, RISE.required_labels)
        # batch axis should always be first
        input_data = utils.move_axis(input_data, 'batch', 0)
        # ensure channels axis is last and keep track of where it was so we can move it back
        channels_axis_index = input_data.dims.index('channels')
        input_data = utils.move_axis(input_data, 'channels', -1)

        p_keep = self._determine_p_keep_for_images(input_data, runner) if self.p_keep is None else self.p_keep

        # data shape without batch axis and channel axis
        img_shape = input_data.shape[1:3]
        # Expose masks for to make user inspection possible
        self.masks = self.generate_masks_for_images(img_shape, p_keep, self.n_masks)

        # Make sure multiplication is being done for correct axes
        masked = (input_data * self.masks)
        # ensure channels axis is in original location again
        masked = utils.move_axis(masked, 'channels', channels_axis_index)
        # convert to numpy for onnx
        masked = masked.values.astype(input_data.dtype)

        batch_predictions = []
        for i in tqdm(range(0, self.n_masks, batch_size), desc='Explaining'):
            batch_predictions.append(runner(masked[i:i + batch_size]))
        self.predictions = np.concatenate(batch_predictions)

        saliency = self.predictions.T.dot(self.masks.reshape(self.n_masks, -1)).reshape(-1, *img_shape)
        return normalize(saliency, self.n_masks, p_keep)

    def _determine_p_keep_for_images(self, input_data, runner, n_masks=100):
        p_keeps = np.arange(0.1, 1.0, 0.1)
        stds = []
        for p_keep in p_keeps:
            std = self._calculate_mean_class_std_for_images(p_keep, runner, input_data, n_masks=n_masks)
            stds += [std]
        best_i = np.argmax(stds)
        best_p_keep = p_keeps[best_i]
        print(f'Rise parameter p_keep was automatically determined at {best_p_keep}')
        return best_p_keep

    def _calculate_mean_class_std_for_images(self, p_keep, runner, input_data, n_masks):
        batch_size = 50
        img_shape = input_data.shape[1:3]
        masks = self.generate_masks_for_images(img_shape, p_keep, n_masks)
        masked = input_data * masks
        predictions = []
        for i in range(0, n_masks, batch_size):
            current_input = masked[i:i + batch_size]
            current_predictions = runner(current_input)
            predictions.append(current_predictions)
        predictions = np.concatenate(predictions)
        std_per_class = predictions.std(axis=0)
        return np.mean(std_per_class)

    def generate_masks_for_images(self, input_size, p_keep, n_masks):
        """Generate a set of random masks to mask the input data

        Args:
            input_size (int): Size of a single sample of input data, for images without the channel axis.
        Returns:
            The generated masks (np.ndarray)
        """
        cell_size = np.ceil(np.array(input_size) / self.feature_res)
        up_size = (self.feature_res + 1) * cell_size

        grid = np.random.choice(a=(True, False), size=(n_masks, self.feature_res, self.feature_res),
                                p=(p_keep, 1 - p_keep))
        grid = grid.astype('float32')

        masks = np.empty((n_masks, *input_size))

        for i in range(n_masks):
            y = np.random.randint(0, cell_size[0])
            x = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            masks[i, :, :] = _upscale(grid[i], up_size)[y:y + input_size[0], x:x + input_size[1]]
        masks = masks.reshape(-1, *input_size, 1)
        return masks
