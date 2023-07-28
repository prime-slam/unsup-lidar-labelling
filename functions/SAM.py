import numpy as np

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def SAM(img_cur, path_to_checkpoint):
    sam = sam_model_registry["vit_h"](checkpoint=path_to_checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks_cur = mask_generator.generate(img_cur)
    return masks_cur


def show_anns_for_SAM(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    return img