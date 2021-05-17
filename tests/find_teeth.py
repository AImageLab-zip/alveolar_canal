import numpy as np
import cv2
from skimage.segmentation import circle_level_set
from visualize_results import MultiView
from matplotlib import pyplot as plt
from skimage.segmentation import active_contour
from skimage.measure import find_contours, perimeter
from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import cc3d
from skimage.morphology import area_closing, binary_erosion
from Jaw import Jaw
from tqdm import tqdm
from mayavi import mlab

GEODESIC_NOISE_TH = 1300  # below this threshold suppress the pixels in the active countour mask
CONFIDENT_TOOTH_TH = 1900  # above this threshold this is teeth for sure
NOISE_DIMLABELS = 1200  # minimum number of elements in a connected component after CONFIDERNT_TOOTH threshold

def plot_2D(image):
    plt.imshow(
        image,
        cmap='gray'
    )
    plt.show()

def plot_3D(volume):
    mlab.contour3d(volume, color=(0.4, 0.9, 0.2))
    mlab.show()

def get_rgb(image):
    rgb = np.stack([image, image, image])
    rgb = np.moveaxis(rgb, 0, -1)
    return rgb

def mask_on_image(img, mask):
    c = np.argwhere(mask)
    if img.ndim == 2:
        img = get_rgb(img)
    img[c[:, 0], c[:, 1]] = (255, 0, 0)
    return img

def plot_contour(mask, image):
    c = find_contours(mask)
    contour = image
    if image.ndim == 2:
        contour = get_rgb(image)
    for cnt in c:
        contour[cnt[:, 0].astype(int), cnt[:, 1].astype(int)] = (255, 0, 0)
    plt.imshow(contour)
    plt.show()

def geodesic(mask, idxs, data, result, confident_mask, iterations=3, baloon=1, desc='something magic is happening!'):

    for idx in tqdm(idxs, desc=desc):

        mask[data[idx] < GEODESIC_NOISE_TH] = 0  # suppress super black new area from the previous
        mask |= confident_mask[idx]  # we add to the mask of this slide all the pixels we are confident to be target

        if np.sum(mask) == 0:
            continue

        mask = morphological_geodesic_active_contour(
            inverse_gaussian_gradient(data[idx] / data[idx].max()),
            iterations,
            mask.astype(bool),
            smoothing=2,
            balloon=baloon
        )
        mask = area_closing(mask)  # removing little squirks inside the masks

        result[idx] = mask  # saving

def find_teeth(jaw, debug=False):
    original_volume = jaw.get_volume()
    original_volume = np.flip(original_volume, 0)
    original_volume = np.clip(original_volume, 0, original_volume.max())  # HU elements under the air th are clipped

    one_third = original_volume.shape[1] // 3
    end_point = int(original_volume.shape[0]/3*1.5)

    teeth_volume = np.zeros_like(original_volume)
    teeth_volume[-end_point:,:-one_third] = original_volume[-end_point:,:-one_third]


    remove_me = np.zeros_like(original_volume)
    # this threashold values from the backward of the jaw which generally belogs to
    # bones on the jaw we dont like. we start from this to propagate the mask all over the jaw bones
    # this mask will be used to avoid our main algorithm to dive into those areas
    tmp = np.zeros_like(original_volume)
    tmp[:,-one_third:] = original_volume[:,-one_third:]
    remove_me[tmp > 1400] = 1

    teeth_volume = np.where(teeth_volume > CONFIDENT_TOOTH_TH, 1, 0)  # we know this are teeth, we just need to expand them properly into the roots
    # filtering noise with labelling
    teeth_volume = cc3d.connected_components(teeth_volume, return_N=False)
    u, count = np.unique(teeth_volume, return_counts=True)
    count[count < NOISE_DIMLABELS] = 0
    count[0] = 0
    labels = u[np.argwhere(count != 0)]
    teeth_volume = np.isin(teeth_volume, labels).astype(np.uint8)

    # best slices to start the propagation is the one with the highest number of teeth
    idx = np.argmax(np.sum(teeth_volume, axis=(1, 2)))
    teeth_img = teeth_volume[idx]


    ####
    # GEODESIC ACTIVE CONTOUR
    # memento: this tends to expand - how to force contraction? just set baloon to a negative value

    tmp = np.zeros_like(remove_me)
    geodesic(
        remove_me[-1].astype(bool),  # let's start from the top
        range(original_volume.shape[0] - 1, 0, -1),
        original_volume,
        tmp,  # we put here most of the jawbones if that works
        remove_me.astype(bool),  # we know those are jawbones
        iterations=4, # we are very explorative!
        baloon=3,  # we are very explorative!
        desc='generating jaw mask'
    )

    if debug: MultiView([(original_volume, 0.4), tmp]).show()
    nojaw = original_volume.copy()
    nojaw[tmp != 0] = 0  # supress the jawbones to reduce mistakes

    result = np.zeros_like(original_volume)
    geodesic(teeth_img.astype(np.int8), range(idx, original_volume.shape[0] - 1), nojaw, result, teeth_volume, desc='generating teeth mask (above)')  # find teeth going above
    geodesic(teeth_img.astype(np.int8), range(idx, 0, -1), nojaw, result, teeth_volume, desc='generating teeth mask (below)')  # find teeth and roots

    result = np.logical_or(binary_erosion(result), teeth_volume).astype(np.uint8)
    if debug: MultiView([(original_volume, 0.4), teeth_volume], [(original_volume, 0.4), result]).show()
    return result

####
# ALTERNATIVE WITH NORMAL ACTIVE CONTOURS

# seeds = []
# numlabels, mask = cv2.connectedComponents(teeth_img.astype(np.int8))
# bounds_mask = np.zeros_like(mask)
# offset = 3
# for l in range(1, numlabels):
#     coords = np.argwhere(mask == l)
#     center = np.mean(coords, axis=0).astype(int)
#     bounds_mask |= circle_level_set(bounds_mask.shape, center, np.max(np.sqrt(np.sum((coords - center)**2, axis=1))).astype(int))
#     seeds.append(center)
#
# for cnt in find_contours(bounds_mask):
#     contour = get_rgb(image)
#     contour[cnt[:, 0].astype(int), cnt[:, 1].astype(int)] = (255, 0, 0)
#     plt.imshow(contour)
#     plt.show()
#
#     snake = active_contour(img / img.max(), cnt, boundary_condition='free')
#
#     refined = get_rgb(image)
#     refined[snake[:, 0].astype(int), snake[:, 1].astype(int)] = (255, 0, 0)
#     plt.imshow(refined)
#     plt.show()
#     break

####
# ATTEMPT TO USE FLOODFIL

# flood fill looks like a bit of shit in this case
# final = np.zeros_like(image)
# for seed in seeds:
#     cv2.floodFill(image, mask=bounds_mask, seedPoint=(seed[1], seed[0]), newVal=100)

if __name__ == '__main__':
    PATIENT_NAME = "P30"
    filepath = fr"Y:\work\datasets\maxillo\VOLUMES\{PATIENT_NAME}"
    jaw = Jaw(os.path.join(filepath, 'DICOM', 'DICOMDIR'))
    find_teeth(jaw, debug=True)
