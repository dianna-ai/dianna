"""Utility functions for specifically for RISE."""


def saliency(predictions, masks, p_keep):
    """RISE saliency: mask-weighted predictions, normalized by number of masks and keep probability.

    The mean prediction is subtracted before weighting and added back after. This leaves the expected
    saliency unchanged, but removes the noise from how often each feature happened to be unmasked,
    which otherwise dominates when the masks barely change the prediction.

    Args:
        predictions (np.ndarray): Model output per mask, shape (n_masks, n_classes)
        masks (np.ndarray): Masks, shape (n_masks, ...)
        p_keep (float): Keep probability used to generate the masks

    Returns:
        Saliency per class, shape (n_classes, n_features) with features the flattened mask axes.
    """
    n_masks = len(masks)
    mean_prediction = predictions.mean(axis=0)
    weighted = (predictions - mean_prediction).T.dot(masks.reshape(n_masks, -1))
    return weighted / n_masks / p_keep + mean_prediction[:, None]
