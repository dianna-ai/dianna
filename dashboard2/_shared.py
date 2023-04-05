import matplotlib.pyplot as plt


def make_rise_plot(
    *,
    image,
    model,
    labels,
    rise_n_masks,
    rise_feat_res,
    rise_unmask_prob,
):
    fig, ax = plt.subplots()
    ax.imshow(image)
    return fig


def make_kernelshap_plot():
    pass


def make_lime_plot():
    pass
