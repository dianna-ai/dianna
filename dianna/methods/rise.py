import numpy as np
from skimage.transform import resize
from tqdm import tqdm

from dianna.utils import get_function


class RISE:
    """
    RISE implementation based on https://github.com/eclique/RISE/blob/master/Easy_start.ipynb
    """

    def __init__(self, n_masks=1000, feature_res=8, p_keep=0.5):
        """RISE initializer.

        Args:
            n_masks (int): Number of masks to generate.
            feature_res (int): Resolution of features in masks.
            p_keep (float): Fraction of image to keep in each mask
        """
        self.n_masks = n_masks
        self.feature_res = feature_res
        self.p_keep = p_keep
        self.masks = None
        self.predictions = None

    def explain_text(self, model_or_function, input_text, batch_size=100):
        runner = get_function(model_or_function)

        tokens = model_or_function.tokenizer(input_text)
        text_shape = len(tokens)
        self.masks = self.generate_masks_for_text(text_shape)  # Expose masks for to make user inspection possible

        # generate sentences with mask
        tokens = np.asarray(tokens)
        tokens_masked = []
        for mask in self.masks:
            tokens_masked.append(tokens[mask])
        sentences = [" ".join(t) for t in tokens_masked]

        preds = []

        for i in tqdm(range(0, self.n_masks, batch_size), desc='Explaining'):
            preds.append(runner(sentences[i:i + batch_size]))
        preds = np.concatenate(preds)
        self.predictions = preds
        saliency = preds.T.dot(self.masks.reshape(self.n_masks, -1)).reshape(-1, text_shape)
        saliency = saliency / self.n_masks / self.p_keep

        # create word and word indices dimension for return values
        word_lengths = [len(t) for t in tokens]
        word_indices = [sum(word_lengths[:i]) + i for i in range(len(tokens))]

        return list(zip(tokens, word_indices, saliency[0], saliency[1]))

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
        runner = get_function(model_or_function)

        # data shape without batch axis and (optional) channel axis
        img_shape = input_data.shape[1:3]
        self.masks = self.generate_masks_for_images(img_shape)  # Expose masks for to make user inspection possible

        predictions = []

        # Make sure multiplication is being done for correct axes
        masked = input_data * self.masks

        for i in tqdm(range(0, self.n_masks, batch_size), desc='Explaining'):
            predictions.append(runner(masked[i:i + batch_size]))
        predictions = np.concatenate(predictions)
        self.predictions = predictions
        saliency = predictions.T.dot(self.masks.reshape(self.n_masks, -1)).reshape(-1, *img_shape)
        saliency = saliency / self.n_masks / self.p_keep
        return saliency

    def generate_masks_for_text(self, input_size):

        masks = np.random.choice(a=(True, False), size=(self.n_masks, input_size), p=(self.p_keep, 1 - self.p_keep))
        return masks

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
            # Random shifts
            y = np.random.randint(0, cell_size[0])
            x = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect', anti_aliasing=False)[y:y + input_size[0],
                                                                                                    x:x + input_size[1]]
        masks = masks.reshape(-1, *input_size, 1)
        return masks
