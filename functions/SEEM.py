import torch

def SEEM(img_cur, axs_cur, path_to_model):
    model = torch.load(path_to_model + "seem_focalt_v2.pt")
    masks_cur = model(img_cur)
    axs_cur.imshow(img_cur)
    axs_cur.imshow(masks_cur)
    return masks_cur