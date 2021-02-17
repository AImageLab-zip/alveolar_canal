import numpy as np


class Plane:

    def __init__(self, plane_z, plane_w):
        """
        create a new (empty) plane of coords, shape of the plane must be declared here
        Args:
            plane_z (Int): z shape of the plane (usually the Z len of the volume to cut)
            plane_w (Int): w shape of the plane (usually the len of the xy set of coordinates)
        """
        self.Z, self.W = plane_z, plane_w
        self.plane = np.empty((3, self.Z, self.W))

    def from_line(self, xy_set):
        """
        load coordinates from a line (duplicating xy values over all the Z axis)
        Args:
            xy_set (numpy array): set of xy values
        """
        if len(xy_set.shape) > 2:
            raise Exception("coords_to_plane: feed this function with just one set of coords per time")

        x_set, y_set = xy_set[:, 0], xy_set[:, 1]
        self.plane = np.stack((
            np.tile(x_set, (self.Z, 1)),
            np.tile(y_set, (self.Z, 1)),
            np.moveaxis(np.tile(np.arange(0, self.Z, dtype=np.float), (x_set.size, 1)), 0, 1)
        ))

    def get_h_axis(self, z_level):
        """
        create a vector parallel to the Z axis of the plane to be used as reference in the rotation
        Args:
            z_level (Int): level of the z axis where our rotation axis has to be placed. if this is not specified the rotation axis
            is placed in the middle of the plane, otherwise we search over the coords in the plane for the closest value and we
            place the rotation axis at that cell of the plane so that the source of the rotation lays there.
        Returns:
        ux, uy, ux (Float): values for each component of the vector
        """

        plane_centre = {
                'z': z_level,
                'w': self.W // 2
            }

        # get the axis from the vectors of differences upon the centre of the plane (ux, uy, uz)
        u = np.array([
            self.plane[0, plane_centre['z'] + 1, plane_centre['w']] - self.plane[
                0, plane_centre['z'], plane_centre['w']],
            self.plane[1, plane_centre['z'] + 1, plane_centre['w']] - self.plane[
                1, plane_centre['z'], plane_centre['w']],
            self.plane[2, plane_centre['z'] + 1, plane_centre['w']] - self.plane[
                2, plane_centre['z'], plane_centre['w']],
        ])
        ux, uy, uz = u / np.linalg.norm(u)  # normalization
        return ux, uy, uz

    def get_w_axis(self, z_level):
        """
        create a vector parallel to the X axis of the plane to be used as reference in the rotation
        Args:
            z_level (Int): level of the z axis where our rotation axis has to be placed. if this is not specified the rotation axis
            is placed in the middle of the plane, otherwise we search over the coords in the plane for the closest value and we
            place the rotation axis at that cell of the plane so that the source of the rotation lays there.
        Returns:
        ux, uy, ux (Float): values for each component of the vector
        """
        plane_centre = {
            'z': z_level,
            'w': self.W // 2
        }
        # get the axis from the vectors of differences upon the centre of the plane (ux, uy, uz)
        u = np.array([
            self.plane[0, plane_centre['z'], plane_centre['w'] + 1] - self.plane[
                0, plane_centre['z'], plane_centre['w']],
            self.plane[1, plane_centre['z'], plane_centre['w'] + 1] - self.plane[
                1, plane_centre['z'], plane_centre['w']],
            self.plane[2, plane_centre['z'], plane_centre['w'] + 1] - self.plane[
                2, plane_centre['z'], plane_centre['w']],
        ])
        ux, uy, uz = u / np.linalg.norm(u)  # normalization
        return ux, uy, uz

    def tilt_x(self, degrees, z_level=None):
        """
        execute a rotation with the following steps:
            1- translate the space to the origin by subtracting the center coords of the plane
            to all the other elements of the plane
            2- if the rotation axis is not laying on the YZ plane or does not lay on the Z axis,
            rotate the plane and the axis to achieve it
            3- tilt the plane around the Z axis with a 3D rotation matrix
            4- perform the reverse of 2
            5- perform the reverse of 3
            further information at http://paulbourke.net/geometry/rotate/
        Args:
            degrees (Int): angle to rotate about in degrees
            z_level (Int): level of the z axis where our rotation axis has to be placed. if this is not specified the rotation axis
            is placed in the middle of the plane, otherwise we search over the coords in the plane for the closest value and we
            place the rotation axis at that cell of the plane so that the source of the rotation lays there.
        """

        if degrees == 0:
            return

        # Z index of the plane should be as close as possible to the avarage canal Z position
        if not z_level:
            z_level = self.Z // 2  # if we have no clues then let's rotate with respect to the middle of the plane
        else:
            z_level = np.abs(self.plane[2] - z_level).argmin() // self.W

        # shifting to the origin
        centres = self.plane[:, z_level, self.W // 2]
        self.plane = np.subtract(self.plane, centres.reshape(3, 1, 1))  # broadcasting is real fancy

        ux, uy, uz = self.get_h_axis(z_level)

        # align to Z
        d = np.sqrt(uy ** 2 + uz ** 2)
        if d != 0:
            axis_align_matr = np.array([
                [1, 0, 0],
                [0, uz / d, -uy / d],
                [0, uy / d, uz / d]
            ])
            self.plane = np.tensordot(self.plane, axis_align_matr, axes=(0, 1))
            self.plane = np.moveaxis(self.plane, 2, 0)

        # rotate
        angle = np.radians(degrees)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        self.plane = np.tensordot(self.plane, rotation_matrix, axes=(0, 1))
        self.plane = np.moveaxis(self.plane, 2, 0)

        if d != 0:
            axis_align_matr = np.array([
                [1, 0, 0],
                [0, uz / d, uy / d],
                [0, -uy / d, uz / d]
            ])
            self.plane = np.tensordot(self.plane, axis_align_matr, axes=(0, 1))
            self.plane = np.moveaxis(self.plane, 2, 0)

        # shifting to the origin
        self.plane = np.add(self.plane, centres.reshape(3, 1, 1))

        # threshold for overflows
        self.plane[2][self.plane[2] >= self.Z] = self.Z - 1

    def tilt_z(self, degrees, z_level=None):
        """
        execute a rotation with the following steps:
            1- translate the space to the origin by subtracting the center coords of the plane
            to all the other elements of the plane
            2- if the rotation axis is not laying on the YX plane or does not lay on the Y axis,
            rotate the plane and the axis to achieve it
            3- tilt the plane around the Y axis with a 3D rotation matrix
            4- perform the reverse of 2
            5- perform the reverse of 3
            further information at http://paulbourke.net/geometry/rotate/
        Args:
            degrees (Int): angle to rotate about in degrees
            z_level (Int): level of the z axis where our rotation axis has to be placed. if this is not specified the rotation axis
            is placed in the middle of the plane, otherwise we search over the coords in the plane for the closest value and we
            place the rotation axis at that cell of the plane so that the source of the rotation lays there.
        """

        if degrees == 0:
            return

        # Z index of the plane should be as close as possible to the avarage canal Z position
        if not z_level:
            z_level = self.Z // 2  # if we have no clues then let's rotate with respect to the middle of the plane
        else:
            z_level = np.abs(self.plane[2] - z_level).argmin() // self.W

        # shifting to the origin
        centres = self.plane[:, z_level, self.W // 2]
        self.plane = np.subtract(self.plane, centres.reshape(3, 1, 1))

        ux, uy, uz = self.get_w_axis(z_level)

        # align to Y axis
        d = np.sqrt(uy ** 2 + ux ** 2)
        if d != 0:
            axis_align_matr = np.array([
                [uy / d, -ux / d, 0],
                [ux / d, uy / d, 0],
                [0, 0, 1]
            ])
            self.plane = np.tensordot(self.plane, axis_align_matr, axes=(0, 1))
            self.plane = np.moveaxis(self.plane, 2, 0)

        # rotate on y to tilt
        angle = np.radians(degrees)
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],

        ])
        self.plane = np.tensordot(self.plane, rotation_matrix, axes=(0, 1))
        self.plane = np.moveaxis(self.plane, 2, 0)

        # go back from the y alignment
        if d != 0:
            axis_align_matr = np.array([
                [uy / d, ux / d, 0],
                [-ux / d, uy / d, 0],
                [0, 0, 1]
            ])
            self.plane = np.tensordot(self.plane, axis_align_matr, axes=(0, 1))
            self.plane = np.moveaxis(self.plane, 2, 0)

        # shifting to the origin
        self.plane = np.add(self.plane, centres.reshape(3, 1, 1))

        self.plane[2][self.plane[2] >= self.Z] = self.Z - 1  # threshold for overflows

    def get_plane(self):
        """
        order of coordinates in the plane are [0] X, [1] Y, [2] Z
        Returns (numpy array): plane of coordinates
        """
        return self.plane.copy()

    def set_plane(self, plane):
        """
        load the data from an existing plane
        Args:
            plane numpy array: plane of coordinates
        """
        self.plane = plane

    @staticmethod
    def empty_like(plane):
        """
        similar to numpy zeros_like, create an empty plane with the same shape of plane
        Args:
            plane (Plane object): use this for retriving plane dimensions
        Returns:
            a new Plane object
        """
        return Plane(plane.Z, plane.W)

    def __getitem__(self, coord_set):
        return self.plane[coord_set]
