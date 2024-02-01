import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class CameraPoseVisualizer:
    def __init__(self, xlim=[-1, 1], ylim=[-1, 1], zlim=[-10, 0], view_mode='none'):
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.view_mode = view_mode
        self.fig = plt.figure(figsize=(5.12, 5.12))
        self.ax = self.fig.add_subplot(projection = '3d')

        # https://matplotlib.org/3.6.0/api/toolkits/mplot3d/view_angles.html#toolkit-mplot3d-view-angles
        # self.ax.view_init(elev=30, azim=45, roll=15)
        if view_mode == 'xz':
            self.ax.view_init(elev=0, azim=-90, roll=0) # XZ
        elif view_mode == 'xy':
            self.ax.view_init(elev=90, azim=-90, roll=0) # XY

        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=0.5, aspect_ratio=0.15):
        # aspect ratio: fov
        # focal_len_scaled: scale 
        self.reset(self.xlim, self.ylim, self.zlim, self.view_mode)
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [
                            [vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]
        ]
        self.ax.add_collection3d(
            # Poly3DCollection(meshes, facecolors=color, linewidths=0.05, edgecolors=color, alpha=0.35))
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))
        # return a ndarray img
        canvas = self.fig.canvas
        canvas.draw()
        width, height = canvas.get_width_height()
        image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image_array = image_array.reshape(height, width, 3)
        return image_array
    
    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        plt.title('Extrinsic Parameters')
        plt.show()
    
    def reset(self, xlim=[-50, 50], ylim=[-50, 50], zlim=[0, 50], view_mode='none'):
        self.__init__(xlim, ylim, zlim, view_mode)