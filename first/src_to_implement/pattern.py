import numpy as np
import matplotlib.pyplot as plt


class Checker(object):
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.zeros([resolution, resolution])

    def draw(self):
        """
        In order to avoid truncated checkerboard patterns,
        we make sure resolution is evenly dividable by 2 * tile_size.
        """
        if self.resolution % (2 * self.tile_size) == 0:
            even = np.zeros([self.tile_size, self.tile_size])
            odd = np.ones([self.tile_size, self.tile_size])
            concat1 = np.concatenate((even, odd), axis=1)
            concat2 = np.concatenate((odd, even), axis=1)
            concat_final = np.concatenate((concat1, concat2), axis=0)
            number_box = int(self.resolution / (2 * self.tile_size))

            self.output = np.copy(np.tile(concat_final, (number_box, number_box)))
            final_output = np.copy(self.output)
        else:
            assert "Error"

        return final_output

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Circle(object):
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.pos_x = position[0]
        self.pos_y = position[1]
        self.output = np.zeros([resolution, resolution])

    def draw(self):
        x_seq = np.array(np.linspace(0, self.resolution, self.resolution), dtype='int')
        y_seq = np.array(np.linspace(0, self.resolution, self.resolution), dtype='int')
        x_grid, y_grid = np.meshgrid(x_seq, y_seq, indexing='xy')
        self.output = (((x_grid - self.pos_x) ** 2) + ((y_grid - self.pos_y) ** 2) - (self.radius ** 2)) < 0

        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Spectrum(object):
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.zeros([resolution, resolution, 3])

    def draw(self):
        self.output[:, :, 0] = np.tile(np.linspace(0, 1, self.resolution), (self.resolution, 1))
        self.output[:, :, 1] = np.transpose(np.tile(np.linspace(0, 1, self.resolution), (self.resolution, 1)))
        self.output[:, :, 2] = np.tile(np.linspace(1, 0, self.resolution), (self.resolution, 1))

        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output)
        plt.show()




