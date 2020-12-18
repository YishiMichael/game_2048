import tkinter as tk
import tkinter.font as tkFont
import numpy as np

import logic


root = tk.Tk()


class Main(object):
    DEFAULT_SHAPE = (4, 4)
    DEFAULT_BASE = 2
    VIEW_DIM = 2
    CELL_SIDE_LENGTH = 50
    LOWEST_LEVEL_BUFF = 10
    BUFF_GAP = 20
    FONT_SIZE = 12
    BACKGROUND_COLOR = "#776e65"
    TILE_COLOR_MAP = [
        "#bbada0",  # empty cell
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
        "#ffffff",  # empty cell
        "#f3d774",
        "#f3d774",
        "#f9f6f2",
    ]
    DISPLAY_FONT = tkFont.Font(
        family="Helvetica", size=FONT_SIZE, weight=tkFont.BOLD
    )

    def __init__(self, root, shape=None, base=None, view_dim=None):
        self.root = root
        self.shape = np.array(shape or self.DEFAULT_SHAPE)
        self.base = base or self.DEFAULT_BASE
        self.view_dim = view_dim or self.VIEW_DIM
        self.dim = self.shape.size
        assert self.dim <= 6
        self.size = np.prod(self.shape)
        self.view_map = np.zeros(self.size, dtype=np.int8)
        assert self.view_dim <= 3
        #self.buttonmap = []
        #self.style = None
        self.frame_width = np.ceil(
            self.dim / self.view_dim - 1
        ) * self.BUFF_GAP + self.LOWEST_LEVEL_BUFF
        self.axes_units = self.get_axes_units()
        self.canvas = self.set_canvas()
        self.tiles = [0] * self.size
        self.texts = [0] * self.size
        self.start_game()

        # score system

    def ravel(self, indices):
        return np.ravel_multi_index(indices, self.shape)

    def unravel(self, index):
        return np.unravel_index(index, self.shape)

    def get_axes_units(self):
        def units_generator(shape):
            current_unit = self.CELL_SIDE_LENGTH + self.LOWEST_LEVEL_BUFF
            yield current_unit
            for dimension_size in shape[:-1]:
                current_unit *= dimension_size
                current_unit += self.BUFF_GAP
                yield current_unit

        return [
            np.fromiter(
                units_generator(self.shape[view_axis::self.view_dim]),
                dtype=np.float64
            )
            for view_axis in range(self.view_dim)
        ]

    def get_cell_anchor_point(self, index):
        indices = self.unravel(index)
        return np.array([
            np.sum(
                indices[view_axis::self.view_dim] * self.axes_units[view_axis]
            )
            for view_axis in range(self.view_dim)
        ]) + self.frame_width

    def get_full_grid_size(self):
        return self.get_cell_anchor_point(self.size - 1) \
            + self.CELL_SIDE_LENGTH + self.frame_width

    def safe_fetch(self, list_obj, val):
        try:
            return list_obj[val]
        except IndexError:
            return list_obj[-1]

    def set_canvas(self):
        self.root.title("Game of 2048")
        full_size = self.get_full_grid_size()
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

    def draw_background_cell(self, index):
        cell_anchor = self.get_cell_anchor_point(index)
        x0, y0 = cell_anchor
        x1, y1 = cell_anchor + self.CELL_SIDE_LENGTH
        xc, yc = cell_anchor + self.CELL_SIDE_LENGTH / 2
        tile_color = self.safe_fetch(self.TILE_COLOR_MAP, 0)
        text_color = self.safe_fetch(self.TEXT_COLOR, 0)
        self.canvas.create_rectangle(
            x0, y0, x1, y1, fill=tile_color
        )

    def draw_tile(self, index):
        val = self.view_map[index]
        assert val
        cell_anchor = self.get_cell_anchor_point(index)
        x0, y0 = cell_anchor
        x1, y1 = cell_anchor + self.CELL_SIDE_LENGTH
        xc, yc = cell_anchor + self.CELL_SIDE_LENGTH / 2
        tile_color = self.safe_fetch(self.TILE_COLOR_MAP, val)
        text_color = self.safe_fetch(self.TEXT_COLOR, val)
        self.tiles[index] = self.canvas.create_rectangle(
            x0, y0, x1, y1, fill=tile_color
        )
        self.texts[index] = self.canvas.create_text(
            xc, yc, text=str(self.base ** val), fill=text_color,
            font=self.DISPLAY_FONT
        )

    def delete_tile(self, index):
        self.canvas.delete(self.tiles[index])
        self.canvas.delete(self.texts[index])
        self.tiles[index] = 0
        self.texts[index] = 0

    def appear_tile(self, index):
        self.draw_tile(index)

    def move_tiles(self, link_data):  # TODO, make animation
        for from_index in link_data.from_index:
            self.delete_tile(from_index)
        for to_index in set(link_data.to_index):
            self.draw_tile(to_index)

    def shift(self, axis, opposite):
        nvm, link_data, score = logic.push_tiles_using_numpy(
            self.view_map.reshape(self.shape),
            self.base, axis=axis, opposite=opposite
        )
        self.view_map = np.fromiter(nvm.flat, np.int8)
        return link_data

    def get_new_num(self):
        choices = np.flatnonzero(self.view_map == 0)
        assert choices.size
        new_num_index = np.random.choice(choices)
        return new_num_index, 1

    def action(self, event):
        char = event.char
        try:
            assert len(char) == 1
            axis = "jiftawlkhgds".index(char) % 6  # TODO, view_dim != 2
        except (AssertionError, ValueError):
            return False
        if axis >= self.dim:
            return False
        opposite = char in "lkhgds"
        old_map = self.view_map
        link_data = self.shift(axis, opposite)
        if np.array_equal(old_map, self.view_map):
            return False
        self.move_tiles(link_data)
        new_num_index, num = self.get_new_num()
        self.view_map[new_num_index] = num
        self.appear_tile(new_num_index)
        return True

    def start_game(self):
        for index in range(self.size):
            self.draw_background_cell(index)
        init_location = np.random.choice(
            np.arange(self.size), 2, replace=False ## init
        )
        for index in init_location:
            self.view_map[index] = 1
            self.appear_tile(index)
        self.root.bind("<Key>", self.action)
        self.root.mainloop()
        

if __name__ == "__main__":
    Main(root, (4, 4), 2)
