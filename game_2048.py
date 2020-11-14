import tkinter as tk
import tkinter.font as tkFont
import numpy as np

import logic


root = tk.Tk()


class Main(object):
    DEFAULT_SHAPE = (4, 4)
    DEFAULT_BASE = 2
    BOX_SIDE_LENGTH = 50
    LOWEST_LEVEL_BUFF = 10
    BUFF_GAP = 20
    FRAME_WIDTH = 40
    FONT_SIZE = 12
    BACKGROUND_COLOR = "#776e65"
    TILE_COLOR_MAP = [
        "#bbada0",  # empty box
        "#eee4da",
        "#ede0c8",
        "#f2b179",
        "#f59563",
        "#f67c5f",
        "#f65e3b",
        "#edcf72",
        "#edcc61",
        "#edc850",
        "#edc53f",
        "#edc22e",
    ]
    TEXT_COLOR = [
        "#ffffff",  # empty box
        "#f3d774",
        "#f3d774",
        "#f9f6f2",
    ]
    DISPLAY_FONT = tkFont.Font(
        family="Helvetica", size=FONT_SIZE, weight=tkFont.BOLD
    )

    def __init__(self, root, shape=None, base=None):
        self.root = root
        self.shape = np.array(shape or self.DEFAULT_SHAPE)
        self.base = base or self.DEFAULT_BASE
        self.dim = self.shape.size
        assert self.dim <= 12
        self.size = np.prod(self.shape)
        self.view_map = np.zeros(self.shape, dtype=np.int8)
        #self.buttonmap = []
        #self.style = None
        self.x_axis_units, self.y_axis_units = self.get_axis_units()
        self.canvas = self.set_canvas()
        self.start_game()

        # score system

    def ravel(self, indices):
        return np.ravel_multi_index(indices, self.shape)

    def unravel(self, index):
        return np.unravel_index(index, self.shape)

    ## Really questionable ... if no function can make this
    def get_tensor_val(self, tensor, indices):
        return tensor.flatten()[self.ravel(indices)]

    def set_tensor_val(self, tensor, indices, val):
        temp = tensor.flatten()
        temp[self.ravel(indices)] = val
        return temp.reshape(tensor.shape)

    def get_box_val(self, indices):
        return self.get_tensor_val(self.view_map, indices)

    def set_box_val(self, indices, val):
        self.view_map = self.set_tensor_val(self.view_map, indices, val)

    def get_axis_units(self):
        # indices = (3, 6, 7, 2, 5, 4)
        # shape = (11, 12, 13, 14, 15, 16)
        # -> [
        #     3 * (11 * (13 * (50 + 10) + 30) + 50) + 7 * (13 * (50 + 10) + 30) + 5 * (50 + 10) + 30,
        #     6 * (12 * (14 * (50 + 10) + 30) + 50) + 2 * (14 * (50 + 10) + 30) + 4 * (50 + 10) + 30
        # ] = [32880, 64950]
        # indices = (3, 6, 7, 2, 5)
        # shape = (11, 12, 13, 14, 15)
        # -> [
        #     3 * (11 * (13 * (50 + 10) + 30) + 50) + 7 * (13 * (50 + 10) + 30) + 5 * (50 + 10) + 30,
        #     6 * (12 * (50 + 10) + 30) + 2 * (50 + 10) + 30
        # ] = [32880, 4650]
        # np.ravel_multi_index((3, 7, 5), (11, 13, 15))
        # -> 3 * (13 * 15) + 7 * 15 + 5
        def units_generator(shape):
            current_unit = self.BOX_SIDE_LENGTH + self.LOWEST_LEVEL_BUFF
            yield current_unit
            for dimension_size in shape[:-1]:
                current_unit *= dimension_size
                current_unit += self.BUFF_GAP
                yield current_unit
        x_axis_units = np.fromiter(
            units_generator(self.shape[::2][::-1]), dtype=np.float64
        )[::-1]
        y_axis_units = np.fromiter(
            units_generator(self.shape[1::2][::-1]), dtype=np.float64
        )[::-1]
        return x_axis_units, y_axis_units

    def get_box_anchor_point(self, indices):
        return np.array([
            np.sum(indices[::2] * self.x_axis_units),
            np.sum(indices[1::2] * self.y_axis_units)
        ]) + self.FRAME_WIDTH

    def safe_fetch(self, list_obj, val):
        try:
            return list_obj[val]
        except IndexError:
            return list_obj[-1]

    def set_canvas(self):
        self.root.title("Game of 2048")
        full_size = self.get_box_anchor_point(self.shape) \
            + self.BOX_SIDE_LENGTH + self.FRAME_WIDTH
        self.root.geometry("{0}x{1}".format(
            int(full_size[0]), int(full_size[1])
        ))
        # TODO, add score area
        canvas = tk.Canvas(
            self.root, width=full_size[0], height=full_size[1],
            bg=self.BACKGROUND_COLOR
        )
        canvas.pack()
        return canvas

    def draw_box(self, indices, val):
        #self.view_map[indices] = val
        box_anchor = self.get_box_anchor_point(indices)
        x0, y0 = box_anchor
        x1, y1 = box_anchor + self.BOX_SIDE_LENGTH
        xc, yc = box_anchor + self.BOX_SIDE_LENGTH / 2
        tile_color = self.safe_fetch(self.TILE_COLOR_MAP, val)
        self.canvas.create_rectangle(
            x0, y0, x1, y1, fill=tile_color,
            tag="box {0}".format(repr(indices))
        )
        if val:
            text_color = self.safe_fetch(self.TEXT_COLOR, val)
            self.canvas.create_text(
                xc, yc, text=str(2 ** val), fill=text_color,
                font=self.DISPLAY_FONT, tag="text {0}".format(repr(indices))
            )

    def draw_tile(self, indices):
        print(indices)
        val = self.get_box_val(indices)
        assert val
        self.draw_box(indices, val)

    def delete_tile(self, indices):
        self.canvas.delete("box {0}".format(repr(indices)))
        self.canvas.delete("text {0}".format(repr(indices)))

    def appear_tile(self, indices):
        self.draw_tile(indices)

    def move_tile(self, from_indices, to_indices, save=True):
        self.delete_tile(from_indices)
        if save:
            self.draw_tile(to_indices)

    def shift(self, axis, opposite):
        self.view_map, *movements, score = logic.push_tiles_using_numpy(
            self.view_map, self.base, axis=axis, opposite=opposite
        )
        movements = list(zip(*movements))
        return movements

    def get_new_num(self):
        choices = np.flatnonzero(self.view_map == 0)
        assert choices.size
        new_num_index = np.random.choice(choices)
        return self.unravel(new_num_index), 1

    def action(self, event, axis, opposite=False):
        old_map = self.view_map
        movements = self.shift(axis, opposite)
        if np.array_equal(old_map, self.view_map):
            return False
        for from_indices, to_indices, save in movements:
            self.move_tile(from_indices, to_indices, save)
        new_num_indices, num = self.get_new_num()
        self.set_box_val(new_num_indices, num)
        self.appear_tile(new_num_indices)
        return True

    def start_game(self):
        for index in range(self.size):
            self.draw_box(self.unravel(index), 0)
        init_location = np.random.choice(
            np.arange(self.size), 4, replace=False ## init
        )
        for index in init_location:
            self.set_box_val(self.unravel(index), 1)
            self.appear_tile(self.unravel(index))
        
        for i in range(self.dim):
            self.root.bind("watfij"[::-1][i], lambda e: self.action(e, i, False))
            self.root.bind("sdghkl"[::-1][i], lambda e: self.action(e, i, True))

        self.root.mainloop()
        

if __name__ == "__main__":
    Main(root, (4, 4), 2)
