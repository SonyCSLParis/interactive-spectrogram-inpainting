import torch

import matplotlib as mpl
# use matplotlib without an X server
# on desktop, this avoids matplotlib windows from popping around
mpl.use('Agg')


def plot_codemap(codemap: torch.Tensor, num_codes: int,
                 cmap: str = 'viridis'):
    """Plot a latent map using a color code

    Arguments:
        codemap (torch.Tensor)
        num_codes (int): the number of available codes
            in the dictionary, this is used to scale the color map
    """
    max_code_value = num_codes - 1
    return plt.matshow(codemap.numpy() / max_code_value,
                       cmap=cmap)
