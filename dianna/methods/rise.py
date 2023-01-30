import numpy as np
from skimage.transform import resize
from tqdm import tqdm
from dianna import utils


def normalize(saliency, n_masks, p_keep):
    """Normalizes salience by number of masks and keep probability."""
    return saliency / n_masks / p_keep


def _upscale(grid_i, up_size):
    return resize(grid_i, up_size, order=1, mode='reflect', anti_aliasing=False)


def _predict_in_batches(masked, runner):
    batch_size = 50
    predictions = []
    for i in range(0, len(masked), batch_size):
        current_input = masked[i:min(i + batch_size, len(masked))]
        current_predictions = runner(current_input)
        predictions.append(current_predictions)
    predictions = np.concatenate(predictions)
    return predictions


class RISEText:
    """RISE implementation for text based on https://github.com/eclique/RISE/blob/master/Easy_start.ipynb."""

    def __init__(self, n_masks=1000, feature_res=8, p_keep=None,
                 preprocess_function=None):
        """RISE initializer.

        Args:
            n_masks (int): Number of masks to generate.
            feature_res (int): Resolution of features in masks.
            p_keep (float): Fraction of input data to keep in each mask (Default: auto-tune this value).
            preprocess_function (callable, optional): Function to preprocess input data with
        """
        self.n_masks = n_masks
        self.feature_res = feature_res
        self.p_keep = p_keep
        self.preprocess_function = preprocess_function
        self.masks = None
        self.predictions = None

    def explain(self, model_or_function, input_text, labels, tokenizer=None,  # pylint: disable=too-many-arguments
                batch_size=100):
        """Runs the RISE explainer on text.

           The model will be called with masked versions of the input text.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained _or_
                                                 the path to a ONNX model on disk.
            input_text (np.ndarray): Text to be explained
            tokenizer: Tokenizer class with tokenize and convert_tokens_to_string methods, and mask_token attribute
            labels (Iterable(int)): Labels to be explained
            batch_size (int): Batch size to use for running the model.

        Returns:
            Explanation heatmap for each class (np.ndarray).
        """
        if tokenizer is None:
            raise ValueError('Please provide a tokenizer to explain_text.')

        runner = utils.get_function(model_or_function, preprocess_function=self.preprocess_function)
        input_tokens = np.asarray(tokenizer.tokenize(input_text))
        num_tokens = len(input_tokens)
        active_p_keep = self._determine_p_keep(input_tokens, runner, tokenizer) if self.p_keep is None else self.p_keep
        input_shape = (num_tokens,)
        self.masks = self._generate_masks(input_shape, active_p_keep,
                                          self.n_masks)  # Expose masks for to make user inspection possible
        masked_sentences = self._create_masked_sentences(input_tokens, self.masks, tokenizer)
        saliencies = self._get_saliencies(runner, masked_sentences, num_tokens, batch_size, active_p_keep)
        return self._reshape_result(input_tokens, labels, saliencies)

    def _determine_p_keep(self, input_text, runner, tokenizer, n_masks=100):
        """See n_mask default value https://github.com/dianna-ai/dianna/issues/24#issuecomment-1000152233."""
        p_keeps = np.arange(0.1, 1.0, 0.1)
        stds = []
        for p_keep in p_keeps:
            std = self._calculate_mean_class_std(p_keep, runner, input_text, tokenizer, n_masks=n_masks)
            stds += [std]
        best_i = np.argmax(stds)
        best_p_keep = p_keeps[best_i]
        print(f'Rise parameter p_keep was automatically determined at {best_p_keep}')
        return best_p_keep

    def _calculate_mean_class_std(self, p_keep, runner, input_text, tokenizer,  # pylint: disable=too-many-arguments
                                  n_masks):
        masks = self._generate_masks(input_text.shape, p_keep, n_masks)
        masked = self._create_masked_sentences(input_text, masks, tokenizer)
        predictions = _predict_in_batches(masked, runner)
        std_per_class = predictions.std(axis=0)
        return np.max(std_per_class)

    def _generate_masks(self, input_shape, p_keep, n_masks):
        masks = np.random.choice(a=(True, False), size=(n_masks,) + input_shape, p=(p_keep, 1 - p_keep))
        return masks

    def _get_saliencies(self, runner, sentences, num_tokens, batch_size, p_keep):  # pylint: disable=too-many-arguments
        self.predictions = self._get_predictions(sentences, runner, batch_size)
        unnormalized_saliency = self.predictions.T.dot(self.masks.reshape(self.n_masks, -1)).reshape(-1, num_tokens)
        return normalize(unnormalized_saliency, self.n_masks, p_keep)

    @staticmethod
    def _reshape_result(input_tokens, labels, saliencies):
        word_indices = list(range(len(input_tokens)))
        return [list(zip(input_tokens, word_indices, saliencies[label])) for label in labels]

    def _get_predictions(self, sentences, runner, batch_size):
        predictions = []
        for i in tqdm(range(0, self.n_masks, batch_size), desc='Explaining'):
            predictions.append(runner(sentences[i:i + batch_size]))
        predictions = np.concatenate(predictions)
        return predictions

    def _create_masked_sentences(self, tokens, masks, tokenizer):
        tokens_masked_list = [
            [token if keep else tokenizer.mask_token for token, keep in zip(tokens, mask)]
            for mask in masks]
        masked_sentences = [tokenizer.convert_tokens_to_string(tokens_masked)
                            for tokens_masked in tokens_masked_list]
        return masked_sentences


class RISEImage:
    """RISE implementation for images based on https://github.com/eclique/RISE/blob/master/Easy_start.ipynb."""

    def __init__(self, n_masks=1000, feature_res=8, p_keep=None,  # pylint: disable=too-many-arguments
                 axis_labels=None, preprocess_function=None):
        """RISE initializer.

        Args:
            n_masks (int): Number of masks to generate.
            feature_res (int): Resolution of features in masks.
            p_keep (float): Fraction of input data to keep in each mask (Default: auto-tune this value).
            axis_labels (dict/list, optional): If a dict, key,value pairs of axis index, name.
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
        self.axis_labels = axis_labels if axis_labels is not None else []

    def explain(self, model_or_function, input_data, labels, batch_size=100):
        """Runs the RISE explainer on images.

           The model will be called with masked images,
           with a shape defined by `batch_size` and the shape of `input_data`.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained _or_
                                                 the path to a ONNX model on disk.
            input_data (np.ndarray): Image to be explained
            batch_size (int): Batch size to use for running the model.
            labels (Iterable(int)): Labels to be explained

        Returns:
            Explanation heatmap for each class (np.ndarray).
        """
        input_data, runner = self._prepare_input_data_and_model(input_data, model_or_function)

        active_p_keep = self._determine_p_keep(input_data, runner) if self.p_keep is None else self.p_keep

        # data shape without batch axis and channel axis
        img_shape = input_data.shape[1:3]
        # Expose masks for to make user inspection possible
        self.masks = self._generate_masks(img_shape, active_p_keep, self.n_masks)

        # Make sure multiplication is being done for correct axes
        masked = input_data * self.masks

        batch_predictions = []
        for i in tqdm(range(0, self.n_masks, batch_size), desc='Explaining'):
            batch_predictions.append(runner(masked[i:i + batch_size]))
        self.predictions = np.concatenate(batch_predictions)

        saliency = self.predictions.T.dot(self.masks.reshape(self.n_masks, -1)).reshape(-1, *img_shape)
        result = normalize(saliency, self.n_masks, active_p_keep)
        if labels is not None:
            result = result[list(labels)]
        return result

    def _prepare_input_data_and_model(self, input_data, model_or_function):
        """Prepares the input data as an xarray with an added batch dimension and creates a preprocessing function."""
        self._set_axis_labels(input_data)
        input_data = utils.to_xarray(input_data, self.axis_labels)
        input_data = input_data.expand_dims('batch', 0)
        input_data, full_preprocess_function = self._prepare_image_data(input_data)
        runner = utils.get_function(model_or_function, preprocess_function=full_preprocess_function)
        return input_data, runner

    def _set_axis_labels(self, input_data):
        # automatically determine the location of the channels axis if no axis_labels were provided
        axis_label_names = self.axis_labels.values() if isinstance(self.axis_labels, dict) else self.axis_labels
        if not axis_label_names:
            channels_axis_index = utils.locate_channels_axis(input_data.shape)
            self.axis_labels = {channels_axis_index: 'channels'}
        elif 'channels' not in axis_label_names:
            raise ValueError("When providing axis_labels it is required to provide the location"
                             " of the channels axis")

    def _determine_p_keep(self, input_data, runner, n_masks=100):
        """See n_mask default value https://github.com/dianna-ai/dianna/issues/24#issuecomment-1000152233."""
        p_keeps = np.arange(0.1, 1.0, 0.1)
        stds = []
        for p_keep in p_keeps:
            std = self._calculate_max_class_std(p_keep, runner, input_data, n_masks=n_masks)
            stds += [std]
        best_i = np.argmax(stds)
        best_p_keep = p_keeps[best_i]
        print(f'Rise parameter p_keep was automatically determined at {best_p_keep}')
        return best_p_keep

    def _calculate_max_class_std(self, p_keep, runner, input_data, n_masks):
        img_shape = input_data.shape[1:3]
        masks = self._generate_masks(img_shape, p_keep, n_masks)
        masked = input_data * masks
        predictions = _predict_in_batches(masked, runner)
        std_per_class = predictions.std(axis=0)
        return np.max(std_per_class)

    def _generate_masks(self, input_size, p_keep, n_masks):
        """Generates a set of random masks to mask the input data.

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

        masks = np.empty((n_masks, *input_size), dtype=np.float32)

        for i in range(n_masks):
            y = np.random.randint(0, cell_size[0])
            x = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            masks[i, :, :] = _upscale(grid[i], up_size)[y:y + input_size[0], x:x + input_size[1]]
        masks = masks.reshape(-1, *input_size, 1)
        return masks

    def _prepare_image_data(self, input_data):
        """Transforms the data to be of the shape and type RISE expects.

        Args:
            input_data (xarray): Data to be explained

        Returns:
            transformed input data, preprocessing function to use with utils.get_function()
        """
        # ensure channels axis is last and keep track of where it was so we can move it back
        channels_axis_index = input_data.dims.index('channels')
        input_data = utils.move_axis(input_data, 'channels', -1)
        # create preprocessing function that puts model input generated by RISE into the right shape and dtype,
        # followed by running the user's preprocessing function
        full_preprocess_function = self._get_full_preprocess_function(channels_axis_index, input_data.dtype)
        return input_data, full_preprocess_function

    def _get_full_preprocess_function(self, channel_axis_index, dtype):
        """Creates a full preprocessing function.

        Creates a preprocessing function that incorporates both the (optional) user's
        preprocessing function, as well as any needed dtype and shape conversions

        Args:
            channel_axis_index (int): Axis index of the channels in the input data
            dtype (type): Data type of the input data (e.g. np.float32)

        Returns:
            Function that first ensures the data has the same shape and type as the input data,
            then runs the users' preprocessing function
        """

        def moveaxis_function(data):
            return utils.move_axis(data, 'channels', channel_axis_index).astype(dtype).values

        if self.preprocess_function is None:
            return moveaxis_function
        return lambda data: self.preprocess_function(moveaxis_function(data))
