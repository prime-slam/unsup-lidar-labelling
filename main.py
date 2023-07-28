import os
import cv2
import time

from imantics import Polygons, Mask

from functions.SAM import SAM, show_anns_for_SAM
from functions.auxiliary import get_all_paths_to_photos, save_img, save_mask_SAM_to_COCO

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Getting paths to photos and models
path_to_project = os.path.abspath(os.curdir)
path_to_photos = path_to_project + "\\photos\\"
path_to_res = path_to_project + "\\result\\"
path_to_model = path_to_project + "\\model\\"

all_path_to_photos, all_path_to_res = get_all_paths_to_photos(path_to_photos)

number_files_to_process = 0
for i in all_path_to_photos:
    number_files_to_process += len(i)

number_files_processed = 1

# Preparing Variables for COCO
images = []
annotations = []
categories = [
    {
        "supercategory": "none",
        "id": 1,
        "name": "image_2"
    }, {
        "supercategory": "none",
        "id": 2,
        "name": "image_3"
    }
]

for num_folder, cur_folder in enumerate(all_path_to_photos):
    if cur_folder == []:
        break
    os.makedirs(all_path_to_res[num_folder][0][0][:-18], exist_ok=True)
    for num_photos_different_cameras, photos_different_cameras in enumerate(cur_folder):
        t = time.time()
        for num_photo, photo in enumerate(photos_different_cameras):
            img = cv2.imread(photo)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = SAM(img, path_to_model + "sam_vit_h_4b8939.pth")
            save_img(img, show_anns_for_SAM(mask), all_path_to_res[num_folder][num_photos_different_cameras][num_photo])

            images.append(
                {
                    "file_name": photos_different_cameras[num_photo][-10:],
                    "height": 376,
                    "width": 1241,
                    "id": photos_different_cameras[num_photo][-10:-4]
                }
            )
            for mask_for_one_item in mask:
                segmentation_for_COCO = {
                        "image_id": photos_different_cameras[num_photo][-10:-4],
                        "id": num_photo
                }
                polygons = Mask(mask_for_one_item).polygons(mask_for_one_item['segmentation'])
                mask_for_one_item['segmentation'] = polygons.points
                segmentation_for_COCO.update(mask_for_one_item)
                annotations.append(segmentation_for_COCO)

        elapsed = time.time() - t

        print(f"Сurrent photo {photos_different_cameras[1][-10:-4]} by folder number {num_folder + 1} - {num_photos_different_cameras + 1}")
        print(f"This circle has passed in time: {elapsed} seconds")
        print(f"Time left: {elapsed * (number_files_to_process - number_files_processed) / 60} minutes")
        number_files_processed += 1

save_mask_SAM_to_COCO(images, annotations, categories, path_to_res)