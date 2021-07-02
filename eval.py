from statistics import mean
import torch
import pathlib
import numpy as np
from skimage import metrics
import os
import pandas as pd
import zipfile

class Eval:
    def __init__(self, loader_config, project_dir, skip_dump=False):
        self.iou_list = []
        self.dice_list = []
        self.config = loader_config
        self.project_dir = project_dir
        self.eps = 1e-06
        self.classes = loader_config['labels']
        self.hausdord_splits = 6
        self.hausdord_verbose = []
        self.hausdorf_list = []
        self.test_ids = []
        self.skip_dump = skip_dump

    def reset_eval(self):
        self.iou_list.clear()
        self.dice_list.clear()
        self.hausdord_verbose = []
        self.hausdorf_list.clear()
        self.test_ids.clear()

    def mean_metric(self, phase):
        if phase not in ["Train", "Validation", "Test", "Final"]:
            raise Exception(f"this phase is not valid {phase}")

        iou, dice, haus = mean(self.iou_list), mean(self.dice_list), max(self.hausdorf_list)

        if phase == "Final":
            excl_dest = os.path.join(self.project_dir, 'logs', 'results.xlsx')
            cols = [f"s{n}" for n in range(self.hausdord_splits - 1)] + ["L entire"] + [f"s{n}" for n in range(self.hausdord_splits - 1)] + ["R entire"]
            df = pd.DataFrame(np.stack(self.hausdord_verbose), columns=cols)

            df.replace([np.inf, 0], -1, inplace=True)
            df = df.loc[:, df.max() > -1]  # removing column with empty hausdorf

            df.insert(0, 'PATIENT', self.test_ids, True)
            df['haus tot'] = np.round(self.hausdorf_list, 2)
            df['IoU'] = np.round(self.iou_list, 2)
            df['dice'] = np.round(self.dice_list, 2)
            df.to_excel(excl_dest, index=False)
            self.save_zip()  # zip volumes with predictions

        self.reset_eval()
        return iou, dice, haus

    def compute_metrics(self, pred, gt, images, names, phase):
        if phase not in ["Train", "Validation", "Test", "Final"]:
            raise Exception(f"this phase is not valid {phase}")

        pred = pred[None, ...] if pred.ndim == 3 else pred
        gt = gt[None, ...] if gt.ndim == 3 else gt
        assert pred.ndim == gt.ndim, f"Gt and output dimensions are not the same before eval. {pred.ndim} vs {gt.ndim}"

        excluded = ['BACKGROUND', 'UNLABELED']
        labels = [v for k, v in self.classes.items() if k not in excluded]  # exclude background from here
        names = names if isinstance(names, list) else [names]
        self.test_ids += names

        for batch_id in range(pred.shape[0]):
            self.iou_list.append(self.iou(pred[batch_id], gt[batch_id], labels))
            self.dice_list.append(self.dice_coefficient(pred[batch_id], gt[batch_id], labels))
            self.hausdorf_list.append(self.hausdorf(pred[batch_id], gt[batch_id], phase))
            if phase == 'Final' and not self.skip_dump:
                self.dump(gt[batch_id], pred[batch_id], images[batch_id], names[batch_id])

    def hausdorf(self, pred, gt, phase, pixel_spacing=0.3):

        if phase == "Final":
            left = []
            right = []

            width = gt.shape[1]
            splits = np.linspace(0, width, self.hausdord_splits).astype(int)
            half = gt.shape[2] // 2

            for i in range(len(splits) - 1):
                left.append(metrics.hausdorff_distance(
                    gt[:, splits[i]:splits[i + 1], :half],
                    pred[:, splits[i]:splits[i + 1], :half]
                ) * pixel_spacing)

                right.append(metrics.hausdorff_distance(
                    gt[:, splits[i]:splits[i + 1], half:],
                    pred[:, splits[i]:splits[i + 1], half:]
                ) * pixel_spacing)

            right.append(metrics.hausdorff_distance(gt[..., half:], pred[..., half:]) * pixel_spacing)
            left.append(metrics.hausdorff_distance(gt[..., :half], pred[..., :half]) * pixel_spacing)
            self.hausdord_verbose.append(np.round(np.concatenate((left, right)).astype(float), 2))

        return metrics.hausdorff_distance(gt, pred) * pixel_spacing

    def iou(self, pred, gt, labels):
        """
        :param image: SHAPE MUST BE (Z, H W) or (BS, Z, H, W)
        :param gt: SHAPE MUST BE (Z, H W) or (BS, Z, H, W)
        :return:
        """
        c_score = []
        for c in labels:
            gt_class_idx = np.argwhere(gt.flatten() == c)
            intersection = np.sum(pred.flatten()[gt_class_idx] == c)
            union = np.argwhere(gt.flatten() == c).size + np.argwhere(pred.flatten() == c).size - intersection
            c_score.append((intersection + self.eps) / (union + self.eps))
        return sum(c_score) / len(labels)

    def dice_coefficient(self, pred, gt, labels):
        c_score = []
        for c in labels:
            gt_class_idx = np.argwhere(gt.flatten() == c)
            intersection = np.sum(pred.flatten()[gt_class_idx] == c)
            dice_union = np.argwhere(gt.flatten() == c).size + np.argwhere(pred.flatten() == c).size
            c_score.append((2 * intersection + self.eps) / (dice_union + self.eps))
        return sum(c_score) / len(labels)

    def dump(self, gt_volume, prediction, images, patient_name):
        save_dir = os.path.join(self.project_dir, 'numpy', f'{patient_name}')
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(save_dir, 'gt.npy'), gt_volume)
        np.save(os.path.join(save_dir, 'pred.npy'), prediction)
        np.save(os.path.join(save_dir, 'input.npy'), images)

    def save_zip(self):
        zipf = zipfile.ZipFile(os.path.join(self.project_dir, 'numpy.zip'), 'w', zipfile.ZIP_DEFLATED)
        for root, dirs, files in os.walk(os.path.join(self.project_dir, 'numpy')):
            for file in files:
                zipf.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file),
                                           os.path.join(os.path.join(self.project_dir))))
        zipf.close()