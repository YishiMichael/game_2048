"use strict";
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
class Core {
    constructor(shape, base, view_dim) {
        this.shape = shape;
        this.base = base;
        this.view_dim = view_dim;
        this.dim = shape.length;
        this.strides = this.get_strides();
        this.size = prod(shape);
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
        this.strides.reverse().forEach((stride) => {
            result.unshift(Math.floor(index / stride));
        });
        return result;
    }
}
class GridRuler extends Core {
    constructor(core) {
        super(core.shape, core.base, core.view_dim);
        this.cell_side_length = 1.0;
        this.lowest_level_buff = 0.2;
        this.buff_gap = 0.4;
        this.frame_width = Math.ceil(this.dim / this.view_dim - 1) * this.buff_gap + this.lowest_level_buff;
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
            current_unit = this.cell_side_length + this.lowest_level_buff;
            axis_val = this.frame_width;
            for (axis = view_axis; axis < this.size; axis += this.view_dim) {
                axis_val += indices[axis] * current_unit;
                current_unit *= this.shape[axis];
                current_unit += this.buff_gap;
            }
            result.push(axis_val);
        }
        return result;
    }
    get_full_grid_size() {
        let result = [];
        let added_width = this.cell_side_length + this.frame_width;
        this.get_cell_anchor_point(this.size - 1).forEach((axis_val) => {
            result.push(axis_val + added_width);
        });
        return result;
    }
}
class Grid extends Core {
    constructor(shape, base, view_dim) {
        super(shape, base, view_dim);
        this.grid = new Array(this.size).fill(0);
        this.grid_ruler = new GridRuler(this);
    }
    get_strides() {
        let result = [1];
        let temp_stride = 1;
        this.shape.forEach((dimension_size) => {
            temp_stride *= dimension_size;
            result.push(temp_stride);
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
        this.strides.reverse().forEach((stride) => {
            result.unshift(Math.floor(index / stride));
        });
        return result;
    }
    move_tiles(axis, opposite) {
        let grid = this.grid;
        let ngrid = new Array(this.size).fill(0);
        let link_data = [];
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
                            temp_indices.forEach((temp_indice) => {
                                link_data.push(new Array(temp_indice, ngrid_p, 1));
                            });
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
                            link_data.push(new Array(temp_indice, exact_i3, 0));
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
                    link_data.push(new Array(temp_indice, exact_i3, 0));
                });
            }
        }
        this.grid = ngrid;
        //return link_data, score; // TODO
    }
}
var core = new Grid([4, 4], 2, 2);
//core.grid = [3, 2, 2, 1, 0, 2, 2, 3, 1, 1, 2, 1, 0, 0, 2, 0];
//console.log(core);
//core.push_tiles(1, true);
//console.log(core);
//console.log(style);
//console.log(style.primaryColor);
