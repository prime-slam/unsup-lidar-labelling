import os
import cv2
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import matplotlib.pyplot as plt
import json
import pycocotools.mask as mask

"""

ПРИКЛАДНЫЕ ФУНКЦИИ

"""


def get_all_paths_to_photos(path_to_photos):
    all_path_to_res = []
    all_path_to_photos = []
    for cur_directory_photos in os.listdir(path_to_photos):
        path_to_cur_dir_photos = path_to_photos + cur_directory_photos + '\\'
        a = []
        a_res = []
        for num_full_path, full_path in enumerate(
                [path_to_cur_dir_photos + i + '\\' for i in os.listdir(path_to_cur_dir_photos)]):
            b = []
            b_res = []
            for cur_photo in os.listdir(full_path):
                b.append(full_path + cur_photo)
                b_res.append("result".join((full_path + cur_photo).split('photos')))
            a.append(b)
            a_res.append(b_res)
        all_path_to_photos.append([photo for photo in zip(*a)])
        all_path_to_res.append([res for res in zip(*a_res)])
    return (all_path_to_photos, all_path_to_res)


def save_img(img, mask, path):
    h = 3.76
    w = 12.41

    fig = plt.figure(frameon=False)
    fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img)
    ax.imshow(mask)
    fig.savefig(path)
    pass

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_mask_SAM_to_COCO(images, annotations, categories, path_to_res):
    dictionary = {
        "images": images,
        "type": "instances",
        "annotations": annotations,
        "categories": categories
    }
    with open(path_to_res + "\\res.json", "w") as outfile:
        json.dump(dictionary, outfile, cls=NumpyEncoder)
    pass



"""

МЕТРИКИ

"""


def mIoU(ground_truth, pred):
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])

    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

    area_of_intersection = i_height * i_width

    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1

    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1

    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection

    iou = area_of_intersection / area_of_union

    return iou


"""

ФУНКЦИИ ДЛЯ SAM

"""


def SAM(img_cur, path_to_model):
    sam = sam_model_registry["vit_h"](checkpoint=path_to_model + "sam_vit_h_4b8939.pth")
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


"""

ФУНКЦИИ ДЛЯ SEEM

"""


def SEEM(img_cur, axs_cur, path_to_model):
    model = torch.load(path_to_model + "seem_focalt_v2.pt")
    masks_cur = model(img_cur)
    axs_cur.imshow(img_cur)
    # axs_cur.imshow(show_anns(masks_cur))
    axs_cur.imshow(img_cur)
    return masks_cur

