"""Utility functions for specifically for RISE."""


def normalize(saliency, n_masks, p_keep):
    """Normalizes salience by number of masks and keep probability."""
    return saliency / n_masks / p_keep
