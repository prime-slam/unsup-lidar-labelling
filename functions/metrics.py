import numpy as np

def mIoU(gt_image, pred_image):
    intersection_x_top_left = np.maximum(gt_image[0], pred_image[0])
    intersection_y_top_left = np.maximum(gt_image[1], pred_image[1])
    intersection_x_bottom_right = np.minimum(gt_image[2], pred_image[2])
    intersection_y_bottom_right = np.minimum(gt_image[3], pred_image[3])

    # Intersection height and width.
    intersection_height = np.maximum(intersection_y_top_left - intersection_y_bottom_right + 1, np.array(0.))
    intersection_width = np.maximum(intersection_x_top_left - intersection_x_bottom_right + 1, np.array(0.))

    area_of_intersection = intersection_height * intersection_width

    # Ground Truth dimensions.
    gt_height = gt_image[3] - gt_image[1] + 1
    gt_width = gt_image[2] - gt_image[0] + 1

    # Prediction dimensions.
    pd_height = pred_image[3] - pred_image[1] + 1
    pd_width = pred_image[2] - pred_image[0] + 1

    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection

    iou = area_of_intersection / area_of_union

    return iou
