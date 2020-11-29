"use strict";
let GAME_FIELD_WIDTH = 400;
let UNIT = "px";
let MIN_FACTOR = 0.5;
let CELL_SIDE_LENGTH = 1.0;
let LOWEST_LEVEL_BUFF = 0.15;
let BUFF_GAP = 0.4;
let FONT_SIZE_RATIO = 0.55;
let FONT_SIZE_DECREMENT_FACTOR = 0.8;
let FONT_SIZE_RATIO_LOWER_BOUND = 0.09;
let BEGIN_TILES = 2;
let LIMIT = 14;
function sum(array) {
    let result = 0;
    array.forEach((value) => {
        result += value;
    });
    return result;
}
function prod(array) {
    let result = 1;
    array.forEach((value) => {
        result *= value;
    });
    return result;
}
function sample(samples) {
    return samples[Math.floor(Math.random() * samples.length)];
}
class Core {
    constructor(shape, base, view_dim) {
        this.shape = shape;
        this.base = base;
        this.view_dim = view_dim;
        this.dim = shape.length;
        this.strides = this.get_strides();
        this.reversed_strides = this.strides;
        this.reversed_strides.reverse();
        this.size = prod(shape);
        this.grid = new Array(this.size).fill(0);
        this.previous_grid = new Array(this.size).fill(0);
        this.game_over = false;
    }
    get_strides() {
        let result = [];
        let temp_stride = 1;
        this.shape.forEach((dimension_size) => {
            result.push(temp_stride);
            temp_stride *= dimension_size;
        });
        return result;
    }
    ravel(indices) {
        let result = 0;
        this.strides.forEach((stride, axis) => {
            result += indices[axis] * stride;
        });
        return result;
    }
    unravel(index) {
        let result = [];
        this.reversed_strides.forEach((stride) => {
            result.unshift(Math.floor(index / stride));
            index = index % stride;
        });
        return result;
    }
    simulate_move(axis, opposite) {
        let grid = this.grid;
        let ngrid = new Array(this.size).fill(0);
        let link_data = new Map();
        let score = 0;
        let abs_minor_step = prod(this.shape.slice(axis + 1));
        let minor_step = opposite ? -abs_minor_step : abs_minor_step;
        let major_step = abs_minor_step * this.shape[axis];
        let i0;
        let i1;
        let begin_index;
        let ngrid_p;
        let hold;
        let count;
        let i2;
        let index;
        let value;
        let exact_i3;
        let temp_indices;
        for (i0 = 0; i0 < this.size; i0 += major_step) {
            for (i1 = i0; i1 < i0 + abs_minor_step; ++i1) {
                begin_index = opposite ? i1 + major_step - abs_minor_step : i1;
                ngrid_p = begin_index;
                hold = 0;
                count = 0;
                temp_indices = [];
                for (i2 = 0; i2 < this.shape[axis]; ++i2) {
                    index = begin_index + i2 * minor_step;
                    value = grid[index];
                    if (value === 0) {
                        continue;
                    }
                    else if (value === hold) {
                        ++count;
                        temp_indices.push(index);
                        if (count === this.base) {
                            ngrid[ngrid_p] = value + 1;
                            link_data.set(ngrid_p, temp_indices);
                            score += Math.pow(this.base, value + 1);
                            ngrid_p += minor_step;
                            hold = 0;
                            count = 0;
                            temp_indices = [];
                        }
                    }
                    else {
                        temp_indices.forEach((temp_indice, i3) => {
                            exact_i3 = ngrid_p + i3 * minor_step;
                            ngrid[exact_i3] = hold;
                            link_data.set(exact_i3, temp_indice);
                        });
                        ngrid_p += count * minor_step;
                        hold = value;
                        count = 1;
                        temp_indices = [index];
                    }
                }
                temp_indices.forEach((temp_indice, i3) => {
                    exact_i3 = ngrid_p + i3 * minor_step;
                    ngrid[exact_i3] = hold;
                    link_data.set(exact_i3, temp_indice);
                });
            }
        }
        let changed = !this.all_equal(ngrid);
        return {
            link_data: link_data,
            new_grid: ngrid,
            changed: changed,
            score: score,
            game_over: false
        };
    }
    move_tiles(axis, opposite) {
        let move_tiles_result = this.simulate_move(axis, opposite);
        if (move_tiles_result.changed) {
            this.previous_grid = this.grid;
            this.grid = move_tiles_result.new_grid;
            let new_tile_index = this.generate_tile_randomly(); // must have available cells
            move_tiles_result.link_data.set(new_tile_index, null);
            move_tiles_result.game_over = this.check_if_game_over();
        }
        return move_tiles_result;
    }
    all_equal(ngrid) {
        for (let index = 0; index < this.size; ++index) {
            if (this.grid[index] !== ngrid[index]) {
                return false;
            }
        }
        return true;
    }
    has_available_cells() {
        for (let value of this.grid) {
            if (!value) {
                return true;
            }
        }
        return false;
    }
    no_matches_available() {
        for (let axis = 0; axis < this.dim; ++axis) {
            for (let opposite = 0; opposite < 2; ++opposite) {
                if (this.simulate_move(axis, Boolean(opposite)).changed) {
                    return false;
                }
            }
        }
        return true;
    }
    check_if_game_over() {
        let self = this;
        return !self.has_available_cells() && self.no_matches_available();
    }
    get_available_cell_indices() {
        let result = [];
        this.grid.forEach((value, index) => {
            if (!value) {
                result.push(index);
            }
        });
        return result;
    }
    generate_random_tile_value() {
        return Math.random() < 0.9 ? 1 : 2;
    }
    generate_tile_randomly() {
        let available_cell_indices = this.get_available_cell_indices();
        if (!available_cell_indices.length) {
            return -1;
        }
        let index = sample(available_cell_indices);
        let value = this.generate_random_tile_value();
        this.grid[index] = value;
        return index;
    }
}
class GridRuler extends Core {
    constructor(shape, base, view_dim) {
        super(shape, base, view_dim);
        this.frame_width = Math.ceil(this.dim / this.view_dim - 1) * BUFF_GAP + LOWEST_LEVEL_BUFF;
        this.full_grid_size = this.get_full_grid_size();
    }
    get_cell_anchor_point(index) {
        let result = [];
        let indices = this.unravel(index);
        let view_axis;
        let current_unit;
        let axis_val;
        let axis;
        for (view_axis = 0; view_axis < this.view_dim; ++view_axis) {
            current_unit = CELL_SIDE_LENGTH + LOWEST_LEVEL_BUFF;
            axis_val = this.frame_width;
            for (axis = view_axis; axis < this.dim; axis += this.view_dim) {
                axis_val += indices[axis] * current_unit;
                current_unit *= this.shape[axis];
                current_unit += BUFF_GAP;
            }
            result.push(axis_val);
        }
        return result;
    }
    get_full_grid_size() {
        let result = [];
        let added_width = CELL_SIDE_LENGTH + this.frame_width;
        this.get_cell_anchor_point(this.size - 1).forEach((axis_val) => {
            result.push(axis_val + added_width);
        });
        return result;
    }
}
class GameGrid extends GridRuler {
    constructor(shape, base, view_dim) {
        super(shape, base, view_dim);
        this.game_container = document.querySelector(".game_container");
        this.grid_container = document.querySelector(".grid_container");
        this.tile_container = document.querySelector(".tile_container");
        this.init_grid();
        this.add_start_tiles();
    }
    get_ratio_string(ratio) {
        return ratio * 100 + "%";
    }
    get_2D_translate_string(translate_list) {
        let translate_str_list = [];
        translate_list.forEach((ratio) => {
            translate_str_list.push(this.get_ratio_string(ratio));
        });
        return "translate(" + translate_str_list.join(", ") + ")";
    }
    get_text_string(value) {
        let power = Math.pow(this.base, value);
        if (power >= Math.pow(10, 9)) {
            let logarithm = Math.log10(this.base) * value;
            let exponent = Math.floor(logarithm);
            return String(Math.pow(10, logarithm - exponent)).slice(0, 6) + "E" + exponent;
        }
        else {
            return String(power);
        }
    }
    set_tile_position_and_size(cell, index) {
        let anchor_point = this.get_cell_anchor_point(index);
        cell.style.transform = this.get_2D_translate_string(anchor_point);
        cell.style.width = this.get_ratio_string(1.0 / this.full_grid_size[0]);
        cell.style.height = this.get_ratio_string(1.0 / this.full_grid_size[1]);
    }
    init_grid() {
        this.game_container.style.width = GAME_FIELD_WIDTH + UNIT;
        this.game_container.style.height = this.full_grid_size[1] / this.full_grid_size[0] * GAME_FIELD_WIDTH + UNIT;
        this.game_container.style.fontSize = GAME_FIELD_WIDTH / this.full_grid_size[0] + UNIT;
        for (let index = 0; index < this.size; ++index) {
            let cell = document.createElement("div");
            this.grid_container.appendChild(cell);
            cell.classList.add("grid_cell");
            this.set_tile_position_and_size(cell, index);
        }
    }
    add_start_tiles() {
        let index;
        for (let i = 0; i < BEGIN_TILES; ++i) {
            index = this.generate_tile_randomly();
            this.add_tile(index, null);
        }
        this.previous_grid = this.grid;
    }
    apply_classes(element, classes) {
        element.setAttribute("class", classes.join(" "));
    }
    clear_container(container) {
        while (container.firstChild) {
            container.removeChild(container.firstChild);
        }
    }
    add_tile(index, previous) {
        let value = this.grid[index];
        let tile = document.createElement("div");
        this.tile_container.appendChild(tile);
        let value_class = "tile_value_" + Math.min(value, LIMIT);
        let classes = ["tile", "tile_position_" + index, value_class];
        this.set_tile_position_and_size(tile, index);
        let self = this;
        if (typeof previous === "number") {
            // Make sure that the tile gets rendered in the previous position first
            window.requestAnimationFrame(() => {
                // Negative numbers stand for temporary tiles
                if (previous < 0) {
                    classes.push("temporary_tile");
                    previous = -previous - 1;
                }
                classes[1] = "tile_position_" + previous;
                self.apply_classes(tile, classes); // Update the position
            });
        }
        else if (previous) {
            classes.push("tile_merged");
            self.apply_classes(tile, classes);
            // Render the tiles that merged
            previous.forEach((merged_index) => {
                self.add_tile(index, -merged_index - 1);
            });
        }
        else {
            classes.push("tile_new");
            self.apply_classes(tile, classes);
        }
        // Modify font size
        let tile_inner = document.createElement("div");
        tile.appendChild(tile_inner);
        tile_inner.classList.add("tile_inner");
        tile_inner.innerHTML = this.get_text_string(value);
        let temp_font_size_ratio = FONT_SIZE_RATIO;
        do {
            tile_inner.style.fontSize = this.get_ratio_string(temp_font_size_ratio);
            temp_font_size_ratio *= FONT_SIZE_DECREMENT_FACTOR;
        } while (tile_inner.scrollWidth > tile_inner.clientWidth && temp_font_size_ratio > FONT_SIZE_RATIO_LOWER_BOUND);
    }
    actuate(axis, opposite) {
        let move_tiles_result = this.move_tiles(axis, opposite);
        if (!move_tiles_result.changed) {
            return false;
        }
        let self = this;
        window.requestAnimationFrame(() => {
            self.clear_container(self.tile_container);
            //let temporary_tile: Element | null;
            //temporary_tile = document.querySelector(".temporary_tiles");
            //while (temporary_tile) {
            //    this.tile_container.removeChild(temporary_tile);
            //    temporary_tile = document.querySelector(".temporary_tiles");
            //}
            // TODO
            move_tiles_result.link_data.forEach((previous, index) => {
                self.add_tile(index, previous);
            });
            if (move_tiles_result.game_over) {
                self.game_over = true;
                console.log("Game over!");
            }
            //self.updateScore(metadata.score);
            //self.updateBestScore(metadata.bestScore);
            //if (metadata.terminated) {
            //    if (metadata.over) {
            //        self.message(false); // You lose
            //    } else if (metadata.won) {
            //        self.message(true); // You win!
            //    }
            //}
        });
        return true;
    }
    withdraw() {
        let self = this;
        window.requestAnimationFrame(() => {
            self.game_over = false;
            self.grid = self.previous_grid;
            self.clear_container(self.tile_container);
            self.previous_grid.forEach((value, index) => {
                if (value) {
                    self.add_tile(index, null);
                }
            });
        });
    }
}
var game_grid = new GameGrid([4, 4], 2, 2);
//window.onresize = function() {
//    let page_width: number = document.documentElement.offsetWidth;
//    let scale_factor: number = page_width / (game_grid.full_grid_size[0] * game_grid.ratio);
//    scale_factor = Math.max(Math.min(scale_factor, 1.0), game_grid.min_factor);
//    game_grid.game_container.style.transform = game_grid.get_scale_string(scale_factor);
//}
// TODO: style -> set(...), clear all typescript compiling warnings
// smaller size
// handle score, game_over
// local storage
//for (let index: number = 1; index < game_grid.size - 3; ++index) {
//    game_grid.add_tile(index, index);
//}
document.addEventListener("keypress", (event) => {
    event.preventDefault();
    if (event.key === "`") {
        game_grid.withdraw();
        game_grid.game_over = false;
    }
    if (game_grid.game_over) {
        return;
    }
    let choice = "klghsdijtfwa".indexOf(event.key);
    let axis = choice % 6;
    let opposite = choice < 6;
    if (choice < 0 || axis >= 2) {
        return;
    }
    game_grid.actuate(axis, opposite);
});
