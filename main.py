from auxiliary import *
from imantics import Polygons, Mask
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import os
from os import listdir
import cv2
import time

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Получаем пути до фотографий и модели
path_to_project = os.path.abspath(os.curdir)
path_to_photos = path_to_project + "\\photos\\"
path_to_res = path_to_project + "\\result\\"
path_to_model = path_to_project + "\\model\\"

all_path_to_photos, all_path_to_res = get_all_paths_to_photos(path_to_photos)

sum = 0
for i in all_path_to_photos:
    sum += len(i)

y = 1

# Подготовка для COCO
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

# Непосредственная склейка фотографий
for num_folder, cur_folder in enumerate(all_path_to_photos):
    if cur_folder == []:
        break
    os.makedirs(all_path_to_res[num_folder][0][0][:-18], exist_ok=True)
    for num_photo, photo in enumerate(cur_folder):
        t = time.time()
        for num_item, item in enumerate(photo):
            img = cv2.imread(item)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = SAM(img, path_to_model)
            save_img(img, show_anns_for_SAM(mask), all_path_to_res[num_folder][num_photo][num_item])

            images.append(
                {
                    "file_name": photo[num_item][-10:],
                    "height": 376,
                    "width": 1241,
                    "id": photo[num_item][-10:-4]
                }
            )
            for i in mask:
                x = {
                        "image_id": photo[num_item][-10:-4],
                        "id": num_item
                }
                polygons = Mask(i).polygons(i['segmentation'])
                i['segmentation'] = polygons.points
                x.update(i)
                annotations.append(x)

        elapsed = time.time() - t

        print(f"Текущая фотография {photo[1][-10:-4]} по счету в папке {num_folder + 1} - {num_photo + 1}")
        print(f"Этот круг прошел за время: {elapsed} сек.")
        print(f"Осталось времени: {elapsed * (i - y) / 60} мин.")
        y += 1

save_mask_SAM_to_COCO(images, annotations, categories, path_to_res)