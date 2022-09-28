"""
This code is borrowed from https://github.com/uzh-rpg/rpg_timelens by https://github.com/ahmetakman and necessary modifications are made on purpose. Some additional explanations are added for clarification.

@Article{Tulyakov21CVPR,
  author        = {Stepan Tulyakov and Daniel Gehrig and Stamatios Georgoulis and Julius Erbach and Mathias Gehrig and Yuanyou Li and
                  Davide Scaramuzza},
  title         = {{TimeLens}: Event-based Video Frame Interpolation},
  journal       = "IEEE Conference on Computer Vision and Pattern Recognition",
  year          = 2021,
}


"""




from wandb import visualize
from common import event #Be careful to have event.py in the same folder with representation.py / be sure the subdirectory named common.

import torch #Instead of having th as  in the original code torch notation prefered.

import math


def to_voxel_grid(event_sequence, nb_of_time_bins=5, remapping_maps=None):
    """Returns voxel grid representation of event steam.

    In voxel grid representation, temporal dimension is
    discretized into "nb_of_time_bins" bins. The events fir
    polarities are interpolated between two near-by bins
    using bilinear interpolation and summed up.

    If event stream is empty, voxel grid will be empty.
    """
    voxel_grid = torch.zeros(nb_of_time_bins,
                          event_sequence._image_height,
                          event_sequence._image_width,
                          dtype=torch.float32,
                          device='cpu')

    voxel_grid_flat = voxel_grid.flatten()

    # Convert timestamps to [0, nb_of_time_bins] range.
    duration = event_sequence.duration()
    start_timestamp = event_sequence.start_time()
    features = torch.from_numpy(event_sequence._features)
    x = features[:, event.X_COLUMN]
    y = features[:, event.Y_COLUMN]
    polarity = features[:, event.POLARITY_COLUMN].float()
    t = (features[:, event.TIMESTAMP_COLUMN] - start_timestamp) * (nb_of_time_bins - 1) / duration
    t = t.float()



    if remapping_maps is not None:
        remapping_maps = torch.from_numpy(remapping_maps)
        x, y = remapping_maps[:,y,x]

    left_t, right_t = t.floor(), t.floor()+1
    left_x, right_x = x.floor(), x.floor()+1
    left_y, right_y = y.floor(), y.floor()+1

    for lim_x in [left_x, right_x]:
        for lim_y in [left_y, right_y]:
            for lim_t in [left_t, right_t]:
                mask = (0 <= lim_x) & (0 <= lim_y) & (0 <= lim_t) & (lim_x <= event_sequence._image_width-1) \
                       & (lim_y <= event_sequence._image_height-1) & (lim_t <= nb_of_time_bins-1)

                # we cast to long here otherwise the mask is not computed correctly
                lin_idx = lim_x.long() \
                          + lim_y.long() * event_sequence._image_width \
                          + lim_t.long() * event_sequence._image_width * event_sequence._image_height

                weight = polarity * (1-(lim_x-x).abs()) * (1-(lim_y-y).abs()) * (1-(lim_t-t).abs())
                voxel_grid_flat.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())

    return voxel_grid
