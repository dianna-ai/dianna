import numpy as np
from skimage.transform import resize
from tqdm import tqdm
from dianna import utils


def normalize(saliency, n_masks, p_keep):
    """Normalizes salience by number of masks and keep probability."""
    return saliency / n_masks / p_keep


def _upscale(grid_i, up_size):
    return resize(grid_i, up_size, order=1, mode='reflect', anti_aliasing=False)


class RISE:
    """RISE implementation based on https://github.com/eclique/RISE/blob/master/Easy_start.ipynb."""
    # axis labels required to be present in input image data
    required_labels = ('channels', )

    def __init__(self, n_masks=1000, feature_res=8, p_keep=None,  # pylint: disable=too-many-arguments
                 axis_labels=None, preprocess_function=None, mask_string="UNKWORDZ"):
        """RISE initializer.

        Args:
            n_masks (int): Number of masks to generate.
            feature_res (int): Resolution of features in masks.
            p_keep (float): Fraction of image to keep in each mask (Default: auto-tune this value).
            axis_labels (dict/list, optional): If a dict, key,value pairs of axis index, name.
                                               If a list, the name of each axis where the index
                                               in the list is the axis index
            preprocess_function (callable, optional): Function to preprocess input data with
            mask_string (str, optional): String to replace masked tokens with (text only)
        """
        self.n_masks = n_masks
        self.feature_res = feature_res
        self.p_keep = p_keep
        self.preprocess_function = preprocess_function
        self.masks = None
        self.predictions = None
        self.axis_labels = axis_labels if axis_labels is not None else []
        self.mask_string = mask_string

    def explain_text(self, model_or_function, input_text, labels=(0,), batch_size=100):
        """Runs the RISE explainer on text.

           The model will be called with masked versions of the input text.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained _or_
                                                 the path to a ONNX model on disk.
            input_text (np.ndarray): Text to be explained
            labels (list(int)): Labels to be explained
            batch_size (int): Batch size to use for running the model.

        Returns:
            Explanation heatmap for each class (np.ndarray).
        """
        runner = utils.get_function(model_or_function, preprocess_function=self.preprocess_function)
        input_tokens = np.asarray(model_or_function.tokenizer(input_text))
        text_length = len(input_tokens)
        active_p_keep = self._determine_p_keep_for_text(input_tokens, runner) if self.p_keep is None else self.p_keep
        input_shape = (text_length,)
        self.masks = self._generate_masks_for_text(input_shape, active_p_keep,
                                                   self.n_masks)  # Expose masks for to make user inspection possible
        sentences = self._create_masked_sentences(input_tokens, self.masks)
        saliencies = self._get_saliencies(runner, sentences, text_length, batch_size, active_p_keep)
        return self._reshape_result(input_tokens, labels, saliencies)

    def _determine_p_keep_for_text(self, input_data, runner, n_masks=100):
        """See n_mask default value https://github.com/dianna-ai/dianna/issues/24#issuecomment-1000152233."""
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
            predictions.append(current_predictions.max(axis=1))
        predictions = np.concatenate(predictions)
        std_per_class = predictions.std()
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
            tokens_masked.append([token if keep else self.mask_string for token, keep in zip(tokens, mask)])
        sentences = [" ".join(t) for t in tokens_masked]
        return sentences

    def explain_image(self, model_or_function, input_data, labels=None, batch_size=100):
        """Runs the RISE explainer on images.

           The model will be called with masked images,
           with a shape defined by `batch_size` and the shape of `input_data`.

        Args:
            model_or_function (callable or str): The function that runs the model to be explained _or_
                                                 the path to a ONNX model on disk.
            input_data (np.ndarray): Image to be explained
            batch_size (int): Batch size to use for running the model.
            labels (tuple): Labels to be explained

        Returns:
            Explanation heatmap for each class (np.ndarray).
        """
        # convert data to xarray
        input_data = utils.to_xarray(input_data, self.axis_labels, RISE.required_labels)
        # add batch axis as first axis
        input_data = input_data.expand_dims('batch', 0)
        input_data, full_preprocess_function = self._prepare_image_data(input_data)
        runner = utils.get_function(model_or_function, preprocess_function=full_preprocess_function)

        active_p_keep = self._determine_p_keep_for_images(input_data, runner) if self.p_keep is None else self.p_keep

        # data shape without batch axis and channel axis
        img_shape = input_data.shape[1:3]
        # Expose masks for to make user inspection possible
        self.masks = self.generate_masks_for_images(img_shape, active_p_keep, self.n_masks)

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

    def _determine_p_keep_for_images(self, input_data, runner, n_masks=100):
        """See n_mask default value https://github.com/dianna-ai/dianna/issues/24#issuecomment-1000152233."""
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
            predictions.append(current_predictions.max(axis=1))
        predictions = np.concatenate(predictions)
        std_per_class = predictions.std()
        return np.mean(std_per_class)

    def generate_masks_for_images(self, input_size, p_keep, n_masks):
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
