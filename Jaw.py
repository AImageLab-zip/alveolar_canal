from dicom_loader import dicom_from_dicomdir
import numpy as np
from pydicom.filereader import read_dicomdir
from pydicom.pixel_data_handlers.numpy_handler import pack_bits
import os
from pathlib import Path
from Plane import Plane
import processing
from pydicom.pixel_data_handlers.util import apply_voi_lut

OVERLAY_ADDR = 0x6004
MIN_QUANTILE = 0.02
MAX_QUANTILE = 0.98


class Jaw:

    def __init__(self, dicomdir_path, flip=True):
        """
        initialize a jaw object from a dicomdir path
        Args:
            dicomdir_path (String): path to the dicomdir file, MUST include the final DICOMDIR,
            flip (Bool): initial flip of the volume and dicom file lists?
        """
        basename = os.path.basename(dicomdir_path)
        if basename.lower() != 'dicomdir':
            raise Exception("ERROR: DICOMDIR PATH HAS TO END WITH DICOMDIR")

        self.dicom_dir = read_dicomdir(os.path.join(dicomdir_path))
        self.filenames, self.dicom_files, self.volume = dicom_from_dicomdir(self.dicom_dir)
        self.Z, self.H, self.W = self.volume.shape
        self.HU_intercept, self.HU_slope = self.__get_HU_rescale_params()

        if flip:  # Z-axis has to be flipped
            self.volume = np.flip(self.volume, 0)
            self.dicom_files.reverse()
            self.filenames.reverse()

        # DEBUG

        for i in range(self.Z):
            self.volume[i] = apply_voi_lut(arr=self.volume[i], ds=self.dicom_files[i])

        # END DEBUG
        # self.__remove_quantiles()
        self.max_value = self.volume.max()
        # self.__normalize()
        self.gt_volume = self.__build_ann_volume()
        self.HU_volume = self.convert_01_to_HU(self.volume)

    def merge_predictions(self, plane, pred):
        """
        insert the predictions inside the volume
        Args:
            plane (3D numpy array or Plane object): plane with coords for the cut
            pred (2D numpy array): binary predicted image to be insert in the ground truth volume
        """
        if type(plane) is Plane:  # get numpy array if plane obj is passed
            plane = plane.get_plane()
        idx = np.argwhere(pred)  # true value of the mask
        self.gt_volume[
            plane[2, idx[:, 1], idx[:, 0]].astype(np.int),
            plane[1, idx[:, 1], idx[:, 0]].astype(np.int),
            plane[0, idx[:, 1], idx[:, 0]].astype(np.int)
        ] = 1

    ############
    # DICOM OPS
    ############

    def __get_HU_rescale_params(self):
        """
        Retrieves RescaleIntercept and RescaleSlope values from DICOM's DataSet

        Returns:
            (int, int): RescaleIntercept and RescaleSlope values
        """
        HU_intercept = self.dicom_files[0].get((0x0028, 0x1052))
        HU_slope = self.dicom_files[0].get((0x0028, 0x1053))
        if HU_intercept is not None and HU_slope is not None:
            return HU_intercept.value, HU_slope.value
        else:
            return -1000, 1

    def __add_overlay(self, ds, overlay_data, overlay_addr, overlay_desc):
        """
        Add annotation overlay at OVERLAY_ADDR

        Args:
            ds (pydicom.dataset.FileDataset): where to add the annotation overlay
            overlay_data (bytes): data for 'OverlayData' field
            overlay_addr (int): address
            overlay_desc (str): description
        """
        ds.add_new((overlay_addr, 0x0010), "US", self.H)
        ds.add_new((overlay_addr, 0x0011), "US", self.W)
        ds.add_new((overlay_addr, 0x0022), "LO", overlay_desc)
        ds.add_new((overlay_addr, 0x0040), "CS", "G")
        ds.add_new((overlay_addr, 0x0050), "SS", [1, 1])
        ds.add_new((overlay_addr, 0x0100), "US", 1)
        ds.add_new((overlay_addr, 0x0102), "US", 0)
        ds.add_new((overlay_addr, 0x3000), "OB", overlay_data)

    def __overwrite_address(self, volume, overlay_addr=OVERLAY_ADDR, overlay_desc="Marker"):
        """
        Overwrites a specific overlay address with given volumetric data.

        Args:
            volume (np.ndarray): volumetric data
            overlay_addr (int): address
        """
        for slice_num in range(len(self.dicom_files)):
            overlay = volume[slice_num].flatten()
            packed_bytes = pack_bits(overlay)
            if len(packed_bytes) % 2:  # padding if needed
                packed_bytes += b'\x00'
            if self.dicom_files[slice_num].get((overlay_addr, 0x3000)) is None:
                self.__add_overlay(self.dicom_files[slice_num], packed_bytes, overlay_addr, overlay_desc)
            else:
                self.dicom_files[slice_num][overlay_addr, 0x3000].value = packed_bytes

    def save_dicom(self, path):
        """
        export the dicom files and the dicomdir to the path folder

        Args:
            path (str): path where dicom files are going to be saved
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        self.dicom_dir.save_as(os.path.join(path, 'DICOMDIR'))
        for i, dicom in enumerate(self.dicom_files):
            dicom.save_as(os.path.join(path, self.filenames[i]))

    ###############
    # CUT FUNCTIONS
    ###############

    def x_slice(self, x_val, cut_gt=False):
        """
        fix the x-axis value and return a 2D view

        Args:
            x_val (int): value to fix
            cut_gt (bool): if true cut over the ground truth volume, if false cut over the volume
        Returns:
            cut (2D numpy array)
        """
        if cut_gt:
            return np.squeeze(self.gt_volume[:, :, x_val])
        else:
            return np.squeeze(self.volume[:, :, x_val])

    def y_slice(self, y_val, cut_gt=False):
        """
        fix the y-axis value and return a 2D view

        Args:
            y_val (int): value to fix
            cut_gt (bool): if true cut over the ground truth volume, if false cut over the volume

        Returns:
            cut (2D numpy array)
        """
        if cut_gt:
            return np.squeeze(self.gt_volume[:, y_val, :])
        else:
            return np.squeeze(self.volume[:, y_val, :])

    def line_slice(self, xy_set, cut_gt=False, interp_fn='bilinear_interpolation', step_fn=None):
        """
        make a slice using a set of xy coordinates.
        if cut_gt is true the cut is performed on the annotated binary volume and the nearest neighbour interpolation
        is used (we just want 0-1 values). if cut_gt is set to False then the cut is performed on the jawbone volume and
        the interpolation methods can be one of the available interpolation functions.
        xy_set can be one or more set of xy coordinates, the function create an image or a volume of cuts automatically.

        Args:
            xy_set (2D or 3D numpy array):
            cut_gt (bool): if true cuts the ground truth image, if false cuts the original volume.
                Possible values are: bilinear_interpolation, bicubic_interpolation
            interp_fn (str): name of the interpolation function

        Returns:
            a 2D or 3D numpy array with the cuts
        """

        if cut_gt:
            interp_fn = lambda x, y: self.gt_volume[:, int(y), int(x)]  # nearest
        else:
            interp_fn = getattr(self, interp_fn)

        if len(xy_set.shape) == 2:  # one xy set or many?
            xy_set = xy_set[np.newaxis]

        h = self.Z  # depth of the volume
        w = max([len(points) for points in xy_set])
        num_cuts = xy_set.shape[0]

        cut = np.zeros((num_cuts, h, w), np.float32)  # result image
        for num_cut in range(num_cuts):
            step_fn is not None and step_fn(num_cut, num_cuts)
            for w_id, (x, y) in enumerate(xy_set[num_cut]):
                if (x - 2) < 0 or (y - 2) < 0 or (x + 2) >= self.W or (y + 2) >= self.H:
                    cut[num_cut, :, w_id] = np.zeros(shape=self.Z)  # fill the array with zeros if overflowing
                else:
                    cut[num_cut, :, w_id] = interp_fn(x, y)  # interpolation

        if not cut_gt:
            cut = cut.astype(np.float32) / self.max_value  # quick 0-1 norm for the data cut

        # fixing possible overflows
        cut[cut > 1] = 1
        cut[cut < 0] = 0

        return np.squeeze(cut)  # clean axis 0 in case of just one cut

    def plane_slice(self, plane, cut_gt=False, interp_fn='trilinear_interpolation'):
        """
        cut the volumes according to a plane of coordinates. the resulting image has the shape of the plane.
        each point of the plane contains the set of zxy coordinates where the function perform the interpolation.

        Args:
            plane (3D numpy array): shape is 3xZxW where W is the len of the xy set of coordinates.
                values are ordered as follow: [0] z coords, [1] x coords, [2] y coords
            cut_gt (bool): if true cuts is performed on the ground truth volume
            interp_fn (string): name of the interpolation function, if cut_gt is True the interp_fn is nearest.

        Returns:
            cut (2D numpy array)
        """

        if type(plane) is Plane:  # get numpy array if plane obj is passed
            plane = plane.get_plane()

        if cut_gt:
            interp_fn = lambda z, x, y: self.gt_volume[int(z), int(y), int(x)]  # nearest
        else:
            interp_fn = getattr(self, interp_fn)

        cut = np.zeros((self.Z, plane.shape[2]))
        for row in range(self.Z):
            for col in range(plane.shape[2]):
                cut[row, col] = interp_fn(plane[2, row, col], plane[0, row, col], plane[1, row, col])  # z, x, y
        return cut

    def create_panorex(self, coords, include_annotations=False):
        """
        Create a 2D panorex image from a set of coordinates on the dental arch

        Args:
            coords (float numpy array): set of coordinates for the cut
            include_annotations (bool): if this flag is set, the panorex image is returned as an RGB
            image where the labels are marked in red

        Returns:
            panorex (numpy array)
        """
        panorex = np.zeros((self.Z, len(coords)), np.float32)
        for idx, (x, y) in enumerate(coords):
            try:
                panorex[:, idx] = self.bilinear_interpolation(x, y)
            except:
                continue
        panorex = panorex.astype(np.float32) / self.max_value  # 0-1 quick normalization

        if include_annotations:
            panorex_gt = np.zeros((self.Z, len(coords)), np.float32)
            for idx, (x, y) in enumerate(coords):
                try:
                    panorex_gt[:, idx] = np.max(
                        self.gt_volume[
                        :,
                        int(np.floor(y)):int(np.floor(y) + 1) + 1,
                        int(np.floor(x)):int(np.floor(x) + 1) + 1
                        ],
                        axis=(1, 2)
                    )
                except:
                    continue
            panorex = processing.grey_to_rgb(panorex)
            idx = np.argwhere(panorex_gt)
            panorex[idx[:, 0], idx[:, 1]] = (1, 0, 0)

        return panorex

    def convert_01_to_HU(self, data):
        return data * self.max_value * self.HU_slope + self.HU_intercept

    def convert_HU_to_01(self, data):
        return (data - self.HU_intercept) / (self.HU_slope * self.max_value)

    ###################
    # GETTERS | SETTERS
    ###################

    def get_slice(self, slice_num):
        return self.volume[slice_num]

    def get_gt_slice(self, slice_num):
        return self.dicom_files[slice_num].overlay_array(OVERLAY_ADDR)

    def get_volume(self, normalized=False):
        if normalized:
            return self.volume.astype(np.float32) / self.max_value
        else:
            return self.volume

    def get_gt_volume(self, labels: list = None):
        if not labels:
            return self.gt_volume
        if np.max(self.gt_volume) in [0, 1]:
            return self.gt_volume
        gt = np.zeros_like(self.gt_volume)
        for label in labels:
            gt += get_mask_by_label(self.gt_volume, label)
        return gt

    def get_HU_volume(self):
        return self.HU_volume

    def get_min_max_HU(self):
        HU_volume = self.get_HU_volume()
        return HU_volume.min(), HU_volume.max()

    def set_volume(self, volume):
        self.volume = volume

    def set_gt_volume(self, volume):
        self.gt_volume = volume

    ################
    # INTERPOLATIONS
    ################

    def bilinear_interpolation(self, x_func, y_func):
        """
        bilinear interpolation between four pixels of the image given a float set of coords
        Args:
            x_func (float): x coordinate
            y_func (float): y coordinate

        Returns:
            (float) interpolated value according to https://en.wikipedia.org/wiki/Bilinear_interpolation
        """

        x1, x2 = int(np.floor(x_func)), int(np.floor(x_func) + 1)
        y1, y2 = int(np.floor(y_func)), int(np.floor(y_func) + 1)
        dx, dy = x_func - x1, y_func - y1
        P1 = self.volume[:, y1, x1] * (1 - dx) * (1 - dy)
        P2 = self.volume[:, y2, x1] * (1 - dx) * dy
        P3 = self.volume[:, y1, x2] * dx * (1 - dy)
        P4 = self.volume[:, y2, x2] * dx * dy
        return P1 + P2 + P3 + P4

    def trilinear_interpolation(self, z_func, x_func, y_func):
        """
        perform a trilinear interpolation, distance between image pixel is always 1 and is omitted
        Args:
            z_func (float): z coordinate
            x_func (float): x coordinate
            y_func (float): y coordinate
        Returns:
            interpolated value according to https://en.wikipedia.org/wiki/Trilinear_interpolation
        """
        # avoid possible overflows
        x_func = self.W - 2 if x_func + 1 >= self.W else x_func
        z_func = self.Z - 2 if z_func + 1 >= self.Z else z_func
        y_func = self.H - 2 if y_func + 1 >= self.H else y_func

        x1, x2 = int(np.floor(x_func)), int(np.floor(x_func) + 1)
        y1, y2 = int(np.floor(y_func)), int(np.floor(y_func) + 1)
        z1, z2 = int(np.floor(z_func)), int(np.floor(z_func) + 1)

        xd, yd, zd = x_func - x1, y_func - y1, z_func - z1
        c11 = self.volume[z1, y1, x1] * (1 - xd) + self.volume[z1, y1, x2] * xd
        c12 = self.volume[z2, y1, x1] * (1 - xd) + self.volume[z2, y1, x2] * xd
        c21 = self.volume[z1, y2, x1] * (1 - xd) + self.volume[z1, y2, x2] * xd
        c22 = self.volume[z2, y2, x1] * (1 - xd) + self.volume[z2, y2, x2] * xd
        c1 = c11 * (1 - yd) + c21 * yd
        c2 = c12 * (1 - yd) + c22 * yd
        c = c1 * (1 - zd) + c2 * zd
        return c

    def cubic_interpolation(self, p0, p1, p2, p3, coord):
        """
        perform cubic interpolation.
        coord must be rescaled between 0 and 1. we use the floating part between the coords for p1 and p2
        Args:
            p0 (numpy array or float): column of values or value for coord x0
            p1 (numpy array or float): column of values or value for coord x1
            p2 (numpy array or float): column of values or value for coord x2
            p3 (numpy array or float): column of values or value for coord x3
            coord: oordinate to interpolate on

        Returns:
            (float) cubic interpolation according to https://www.paulinternet.nl/?page=bicubic
        """
        if coord == 0:
            return p1  # if we already have an int coord we don't need to interpolate this stripe
        return p1 + 0.5 * coord * (
                p2 - p0 + coord * (2 * p0 - 5 * p1 + 4 * p2 - p3 + coord * (3. * (p1 - p2) + p3 - p0)))

    def bicubic_interpolation(self, x_func, y_func):
        """
        perform bicubic interpolation by firstly first interpolating
        the four columns and then interpolating the results in the y direction
        Args:
            x_func (float numpy array):  x coord to interpolate on
            y_func (float numpy array):  y coord to interpolate on
        Returns:
        (float) all the interpolated values on a z column
        """

        x0, x1, x2, x3 = int(np.floor(x_func)) - 1, int(np.floor(x_func)), int(np.ceil(x_func)), int(
            np.ceil(x_func)) + 1
        y0, y1, y2, y3 = int(np.floor(y_func)) - 1, int(np.floor(y_func)), int(np.ceil(y_func)), int(
            np.ceil(y_func)) + 1

        iy = []
        for y in [y0, y1, y2, y3]:
            i0 = self.cubic_interpolation(
                self.volume[:, y, x0],
                self.volume[:, y, x1],
                self.volume[:, y, x2],
                self.volume[:, y, x3],
                x_func - int(x_func)
            )
            iy.append(i0)
        return self.cubic_interpolation(*iy, y_func - int(y_func))

    def bicubic_interpolation_3d(self, z_func, x_func, y_func):
        """
        perform bicubic interpolation by firstly first interpolating over z,
        then over x and then interpolating the results in the y direction
        Args:
            z_func (float): z coord to interpolate on
            x_func (float): x coord to interpolate on
            y_func (float): y coord to interpolate on

        Returns:
        (float numpy array) All the interpolated values on a z column
        """
        z0, z1, z2, z3 = int(np.floor(z_func)) - 1, int(np.floor(z_func)), int(np.ceil(z_func)), int(
            np.ceil(z_func)) + 1
        x0, x1, x2, x3 = int(np.floor(x_func)) - 1, int(np.floor(x_func)), int(np.ceil(x_func)), int(
            np.ceil(x_func)) + 1
        y0, y1, y2, y3 = int(np.floor(y_func)) - 1, int(np.floor(y_func)), int(np.ceil(y_func)), int(
            np.ceil(y_func)) + 1

        # TODO: avoid overflow, can we do better here?
        if z3 >= self.volume.shape[0]:
            z3 = self.volume.shape[0] - 1

        iy = []
        for z in [z0, z1, z2, z3]:
            ix = []
            for y in [y0, y1, y2, y3]:
                ix0 = self.cubic_interpolation(
                    self.volume[z, y, x0],
                    self.volume[z, y, x1],
                    self.volume[z, y, x2],
                    self.volume[z, y, x3],
                    x_func - int(x_func)
                )
                ix.append(ix0)
            iy.append(self.cubic_interpolation(*ix, y_func - int(y_func)))
        return self.cubic_interpolation(*iy, z_func - int(z_func))

    ###############
    # PRIVATE UTILS
    ###############

    def __remove_quantiles(self, min=MIN_QUANTILE, max=MAX_QUANTILE):
        """
        remove peak values
        Args:
            min (float): min threshold
            max (float): max threshold
        """
        min = np.quantile(self.volume, min),
        max = np.quantile(self.volume, max)
        self.volume[self.volume > max] = max
        self.volume[self.volume < min] = min

    def __normalize(self, type='simple'):
        """
        perform normalizations on the volume
        Args:
            type (String): type of normalizations, simple [0-1]
        """
        if type == 'simple':
            self.max_value = self.volume.max()
            self.volume = self.volume.astype(np.float32) / self.volume.max()

    def __build_ann_volume(self):
        """
        read overlay data from the dicom files and extract them in a numpy array. if
        no annotations are found a 0-volume is created
        """
        annotations = []
        try:
            for slice_num in range(self.Z):
                annotations.append(self.get_gt_slice(slice_num))
            return np.stack(annotations).astype(np.uint8)
        except:
            # print("INFO: NO ANNOTATION FOUND IN THIS VOLUME! BLACK MASK RETURNED")
            return np.zeros_like(self.volume)
