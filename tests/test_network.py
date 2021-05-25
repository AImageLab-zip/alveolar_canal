import yaml
import cv2
import numpy as np
import processing
from Jaw import Jaw
import imageio
from matplotlib import pyplot as plt
import pathlib
from mayavi import mlab
import os
from tqdm import tqdm
from tests.find_teeth import find_teeth


def write_text(image, text, position, size=0.4, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(image, text, position,
                font, size,
                (255, 255, 255, 255),  # font color
                1  # font stroke
    )
    return image


def snapshot(jaw, title, teeth):
    fig = mlab.figure(size=(700,700))

    gt = np.swapaxes(jaw.get_gt_volume(), 0, 2)
    data = np.swapaxes(jaw.get_volume(normalized=True), 0, 2)
    mlab.contour3d(data, color=(0.5, 0.5, 0.5), opacity=.3, contours=3,  figure=fig)
    mlab.contour3d(gt, color=(1, 0, 0), figure=fig)
    if teeth is not None:
        teeth = np.swapaxes(teeth, 0, 2)
        mlab.contour3d(teeth, color=(0, 0, 0), figure=fig)
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
    dir_config = config.get('directories')

    patient_folders = dir_config['test_folders']  # folder names of the used test set
    dataset_folders = dir_config['dataset_folder']  # load dicom for sparse annotation + check network input was correct
    source_path = dir_config['data_folder']  # load prediction, input and gt from network saves

    folder_name = str(config['experiment'].get('title', 'unknown'))
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

    do_panorex = config['output'].get('panorex', True)
    do_sides = config['output'].get('side_cuts', True)
    do_3D = config['output'].get('volume_overview', True)
    include_teeth = config['output'].get('volume_overview', True)

    for i, (folder) in enumerate(patient_folders):

        print(f"going for patient {folder}")
        data = np.squeeze(np.load(os.path.join(source_path, str(folder), 'input.npy')))
        gt = np.squeeze(np.load(os.path.join(source_path, str(folder), 'gt.npy')))
        pred = np.squeeze(np.load(os.path.join(source_path, str(folder), 'pred.npy')))

        jaw = Jaw(os.path.join(dataset_folders, str(folder), 'DICOM', 'DICOMDIR'))

        assert np.all(np.squeeze(gt) == np.load(os.path.join(dataset_folders, str(folder), 'gt_alpha.npy'))), 'gt from experiment does not match with the dataset one'

        # first we use the panorex from the 2D original annotation to get a spline
        coords = np.argwhere(jaw.get_gt_volume())
        x, y = coords[:, 2], coords[:, 1]
        pol = np.polyfit(x, y, 12)
        p = np.poly1d(pol)
        loff, coords, hoff, der = processing.arch_lines(p, min(x), max(x), offset=50)

        # we generate side coords for the second test
        side_coords = processing.generate_side_coords(hoff, loff, der, offset=100)

        pathlib.Path(os.path.join(folder_name, str(folder))).mkdir(parents=True, exist_ok=True)

        teeth = None
        if include_teeth:
            teeth = find_teeth(jaw)
            teeth = np.flip(teeth, 0)

        # PANOREX AND SIDE CUTS FROM THE ORIGINAL ANNOTATION
        panorex = []
        side_cuts = []
        views = []
        for i, (test_vol, title) in enumerate([(jaw.get_gt_volume(), '2D annotation'), (gt, '3D annotation'), (pred, 'our model')]):
            jaw.set_gt_volume(test_vol)
            if do_panorex:
                panoramic = jaw.create_panorex(coords, include_annotations=True)
                panorex.append(cv2.normalize(panoramic, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

            if do_sides:
                sides = processing.grey_to_rgb(jaw.line_slice(side_coords))
                gt_coords = np.argwhere(jaw.line_slice(side_coords, cut_gt=True))
                sides[gt_coords[:, 0], gt_coords[:, 1], gt_coords[:, 2], :] = (1, 0, 0)
                side_cuts.append(sides)

            if do_3D:
                views.append(snapshot(jaw, title, teeth))

        if do_panorex:
            panorex = np.concatenate(panorex)
            plt.axis('off')
            plt.imshow(panorex)
            plt.savefig(os.path.join(folder_name, str(folder), 'panorex.png'), bbox_inches='tight')

        if do_sides:
            # converting to uint8 for the video
            side_cuts = np.concatenate(side_cuts, axis=2)
            side_cuts *= 255.0 / side_cuts.max()
            side_cuts = side_cuts.astype(np.uint8)

            # writing the labels on the video
            banner = np.zeros((30, side_cuts.shape[2], 3), dtype=np.uint8)
            write_text(banner, "old labels", (5, 20))
            write_text(banner, "new labels", (side_cuts.shape[2] // 3, 20))
            write_text(banner, "our model", (2 * side_cuts.shape[2] // 3, 20))
            banner = np.repeat(banner[np.newaxis], side_cuts.shape[0], 0)
            side_cuts = np.concatenate((banner, side_cuts), axis=1)

            # "end" banner for the video
            end_message = np.zeros((side_cuts.shape[1:]), dtype=np.uint8)
            write_text(end_message, "fin.",
                       position=(100, 20 + end_message.shape[0] // 2),
                       size=3, font=cv2.FONT_HERSHEY_SCRIPT_COMPLEX
                       )
            end_message = np.repeat(end_message[np.newaxis], 20, 0)
            side_cuts = np.concatenate((side_cuts, end_message), axis=0)

            # finally save the data as a video
            w = imageio.get_writer(os.path.join(folder_name, str(folder), 'side_cuts.mp4'), format='FFMPEG', fps=10)
            for s in side_cuts:
                w.append_data(s)
            w.close()

        # 3D views
        if do_3D:
            views = np.concatenate(views, axis=2)
            w = imageio.get_writer(os.path.join(folder_name, str(folder), 'rotation.mp4'), format='FFMPEG', fps=30)
            for v in views:
                w.append_data(v)
            w.close()
