import json
import matplotlib.pyplot as plt
import numpy as np
import os
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_all_paths_to_photos(path_to_photos):
    all_path_to_res = []
    all_path_to_photos = []
    for cur_dir_photos in os.listdir(path_to_photos):
        path_to_cur_dir_photos = path_to_photos + cur_dir_photos + '\\'
        all_paths_separated_into_image_folders = []
        all_res_paths_separated_into_image_folders = []
        for num_full_path, full_path in enumerate(
                [path_to_cur_dir_photos + i + '\\' for i in os.listdir(path_to_cur_dir_photos)]):
            all_paths_cur_image_folder = []
            all_res_paths_cur_image_folder = []
            for cur_photo in os.listdir(full_path):
                all_paths_cur_image_folder.append(full_path + cur_photo)
                all_res_paths_cur_image_folder.append("result".join((full_path + cur_photo).split('photos')))
            all_paths_separated_into_image_folders.append(all_paths_cur_image_folder)
            all_res_paths_separated_into_image_folders.append(all_res_paths_cur_image_folder)
        all_path_to_photos.append([photo for photo in zip(*all_paths_separated_into_image_folders)])
        all_path_to_res.append([res for res in zip(*all_res_paths_separated_into_image_folders)])
    return (all_path_to_photos, all_path_to_res)


def save_img(img, mask, path):
    # Source photo parameters in inches
    # size_in_inches = size_in_pixel / rcParams['figure.dpi'], rcParams['figure.dpi'] = 100 (by default)

    h = 3.76
    w = 12.41

    fig = plt.figure(frameon=False)
    fig.set_size_inches(w, h)
    # [x0, y0, width, height] denoting the lower left point of the new axes in figure coodinates and its width and height.
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img)
    ax.imshow(mask)
    fig.savefig(path)

def save_mask_SAM_to_COCO(images, annotations, categories, path_to_res):
    coco_data = {
        "images": images,
        "type": "instances",
        "annotations": annotations,
        "categories": categories
    }
    with open(path_to_res + "\\res.json", "w") as outfile:
        json.dump(coco_data, outfile, cls=NumpyEncoder)