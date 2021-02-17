import yaml
import cv2
import numpy as np
import processing
from Jaw import Jaw
import imageio
from matplotlib import pyplot as plt
import pathlib
from mayavi import mlab


def write_text(image, text, position, size=0.4, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(image, text, position,
                font, size,
                (255, 255, 255, 255),  # font color
                1  # font stroke
    )
    return image


def snapshot(jaw, title):
    fig = mlab.figure(size=(700,700))
    gt = np.swapaxes(jaw.get_gt_volume(), 0, 2)
    data = np.swapaxes(jaw.get_volume(normalized=True), 0, 2)
    mlab.contour3d(data, color=(0.5, 0.5, 0.5), opacity=.3, contours=3,  figure=fig)
    mlab.contour3d(gt, color=(1, 0, 0), figure=fig)
    mlab.text(color=(0, 0, 0), y=.9, x=0.4, width=.3, text=title, figure=fig)

    angles = np.arange(0, 360)
    rolls = np.full_like(angles, -90)
    rolls[:90] = 90
    rolls[270:] = -270
    params = np.stack((angles, rolls))
    params = np.delete(params, (90, 270), axis=1)

    rotation = []
    for angle, roll in np.rollaxis(params, 1):

        mlab.view(
            focalpoint=(gt.shape[0] // 2, gt.shape[1] // 2, gt.shape[2] // 2),
            distance=800, roll=roll, elevation=90, azimuth=angle
        )
        snap = mlab.screenshot(figure=fig)
        while(snap.shape[0] < 50):
            snap = mlab.screenshot(figure=fig)
        rotation.append(snap)
    mlab.close(fig)

    return np.stack(rotation)


if __name__ == '__main__':
    config = yaml.safe_load(open('test_config.yaml', 'r'))

    jaw = Jaw(config['directories'].get('dicom'))
    gt = np.load(config['directories'].get('gt'))

    # qui mettiamo la parte dello split, preprocessing lo importiamo direttamente da utils appena avrà senso
    # input va scalato e paddato
    # carichiamo rete e pesi e prediciamo pred

    pred = np.load(r'C:\Users\marco\Desktop\test\volumi\133_pred.npy')

    # first we use the panorex from the 2D original annotation to get a spline
    coords = np.argwhere(jaw.get_gt_volume())
    x, y = coords[:, 2], coords[:, 1]
    pol = np.polyfit(x, y, 12)
    p = np.poly1d(pol)
    loff, coords, hoff, der = processing.arch_lines(p, min(x), max(x), offset=50)

    # we generate side coords for the second test
    side_coords = processing.generate_side_coords(hoff, loff, der, offset=100)

    folder_name = str(config['experiment'].get('patient', 'unknown'))
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

    # PANOREX AND SIDE CUTS FROM THE ORIGINAL ANNOTATION
    panorex = []
    side_cuts = []
    views = []
    for i, (test_vol, title) in enumerate([(jaw.get_gt_volume(), '2D annotation'), (gt, '3D annotation'), (pred, 'our model')]):
        jaw.set_gt_volume(test_vol)
        panoramic = jaw.create_panorex(coords, include_annotations=True)
        panorex.append(cv2.normalize(panoramic, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

        sides = processing.grey_to_rgb(jaw.line_slice(side_coords))
        gt_coords = np.argwhere(jaw.line_slice(side_coords, cut_gt=True))
        sides[gt_coords[:, 0], gt_coords[:, 1], gt_coords[:, 2], :] = (1, 0, 0)
        side_cuts.append(sides)

        views.append(snapshot(jaw, title))


    panorex = np.concatenate(panorex)
    plt.axis('off')
    plt.imshow(panorex)
    plt.savefig(f'{folder_name}/panorex', bbox_inches='tight')

    # converting to uint8 for the gif
    side_cuts = np.concatenate(side_cuts, axis=2)
    side_cuts *= 255.0 / side_cuts.max()
    side_cuts = side_cuts.astype(np.uint8)

    # writing the labels on the gif
    banner = np.zeros((30, side_cuts.shape[2], 3), dtype=np.uint8)
    write_text(banner, "old labels", (5, 20))
    write_text(banner, "new labels", (side_cuts.shape[2] // 3, 20))
    write_text(banner, "our model", (2 * side_cuts.shape[2] // 3, 20))
    banner = np.repeat(banner[np.newaxis], side_cuts.shape[0], 0)
    side_cuts = np.concatenate((banner, side_cuts), axis=1)

    # "end" banner for the gif
    end_message = np.zeros((side_cuts.shape[1:]), dtype=np.uint8)
    write_text(end_message, "fin.",
               position=(100, 20 + end_message.shape[0] // 2),
               size=3, font=cv2.FONT_HERSHEY_SCRIPT_COMPLEX
               )
    end_message = np.repeat(end_message[np.newaxis], 20, 0)
    side_cuts = np.concatenate((side_cuts, end_message), axis=0)

    # finally save the data as a gift
    imageio.mimsave(f'{folder_name}/side_cuts.gif', [s for s in side_cuts])

    # 3D views
    views = np.concatenate(views, axis=2)
    w = imageio.get_writer(f'{folder_name}/rotation.mp4', format='FFMPEG', fps=30)
    for v in views:
        w.append_data(v)
    w.close()
