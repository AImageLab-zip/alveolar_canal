from mayavi import mlab
from mayavi.core.ui.api import MlabSceneModel, SceneEditor
from traits.api import HasTraits
from traitsui.api import View, Item, HSplit, VSplit, VFold
import numpy as np
from matplotlib import pyplot as plt


def simple_viewer(image):
    plt.imshow(image, cmap='gray')
    plt.show()


COLORS = [
    (0.4, 0.9, 0.2), (0.8, 0.1, 0.2), (0.1, 0.1, 0.4),
    (0.6, 1, 0.1), (0, 0, 0), (1, 1, 1), (0, 0.4, 0.3),
    (0.1, 0.4, 0.2), (0.1, 1, 0.7), (0.2, 0.9, 0.1),
    (0.3, 0, 0.3), (0.3, 0.9, 1), (0.4, 0, 0.7),
    (0.5, 0, 1), (0.6, 0.3, 0.7)
]


class MultiView(HasTraits):

    def __init__(self, *args):
        """
        Use this class for plotting multiple 3D volumes in multiple windows.
        :param args (List): each element in the list is treated as a new mayavi subwindows which is dinamically generated.
        each element of the list can be either a numpy volume or a list of numpy volumes.
        if a list of volumes is specified, then that volumes will be plot on the same subwindow.
        example:
            MultiView([volume_1, volume_2, volume_3])
            -> 3 sub-windows with a single volume each
            MultiView([[vol_1, detail], [vol_2]])
            -> 2 sub-windows, one of them with two volumes overlapped

        you can also specify an opacity for a given volume by passing a tuple (volume, opacity) instead of a volume
        example:
            MultiView([[(volume_1, 0.2), volume_2], [volume_3]])
            -> 2 sub-windows: one window with two volumes, one of which with 20% opacity and another window with a single volume

        if not specified, opacity is set to 1.
        use show() method for plotting.
        """

        HasTraits.__init__(self)
        windows = [arg for arg in args if isinstance(arg, list)]  # filter not-list object

        self.num_views = len(windows)
        num_colors = max([len(w) for w in windows])  # which is the windows with highest number of volumes?
        if num_colors > 15:
            raise Exception("seriously man? more than 15 plots on a window? i dont' even have enough colours.")

        istance_ids = []  # items will be attached to the mayavi scene by the name/ID of their attributes
        for i, w in enumerate(windows):
            istance_id = 'obj{}'.format(str(i))
            setattr(self, istance_id, MlabSceneModel())
            istance_ids.append(istance_id)

        n_rows = np.floor(np.sqrt(self.num_views)).astype(int)
        n_cols = self.num_views // n_rows
        final_elems = self.num_views % n_rows

        self.items = [Item(w, editor=SceneEditor(), width=1/n_cols) for w in istance_ids]  # one Item per window

        hsplit = []
        for n_row in range(n_rows):
            hsplit.append(HSplit(*self.items[n_row * n_cols:(n_row+1) * n_cols], label='block{}'.format(n_row)))
        if final_elems != 0:
            hsplit.append(HSplit(*self.items[-final_elems:], label='last'))
        self.view = View(VFold(*hsplit), resizable=True, height=0.8, width=0.9)

        for w_id, w in enumerate(windows):
            for v_id, volume in enumerate(w):
                if isinstance(volume, tuple):
                    vol, opacity = volume[0], volume[1]
                else:
                    vol, opacity = volume, 1
                mlab.contour3d(vol, color=tuple(COLORS[v_id]), opacity=opacity, figure=getattr(self, istance_ids[w_id]).mayavi_scene)

    def show(self):
        self.configure_traits(view=self.view)


if __name__ == '__main__':
    tests = [[vol] for vol in np.random.random((7, 30, 30, 30))]
    MultiView(*tests).show()
