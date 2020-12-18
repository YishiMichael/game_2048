"use strict";
function query_selector(selector) {
    return document.querySelector(selector);
}
function $(selector) {
    // Throw an error if no elements are selected
    let element = query_selector(selector);
    if (!element) {
        throw `No elements are selected by "${selector}"`;
    }
    return element;
}
function copy(array) {
    let result = [];
    array.forEach((val) => {
        result.push(val);
    });
    return result;
}
function prod(array) {
    let result = 1;
    array.forEach((val) => {
        result *= val;
    });
    return result;
}
function sample(samples) {
    return samples[Math.floor(Math.random() * samples.length)];
}
class GiantInteger {
    constructor(val = 0) {
        if (typeof val === "number") {
            this.q = [];
            let num = val;
            while (num) {
                this.q.unshift(num % 10);
                num = Math.floor(num / 10);
            }
        }
        else {
            this.q = val;
        }
    }
    g_copy() {
        return new GiantInteger(this.q);
    }
    g_sum(g) {
        if (!this.q.length) {
            return g;
        }
        if (!g.q.length) {
            return this;
        }
        let result = [];
        let carry = 0;
        let l1 = copy(this.q);
        let l2 = copy(g.q);
        let val1;
        let val2;
        let digit;
        while (l1.length || l2.length) {
            val1 = l1.pop() || 0;
            val2 = l2.pop() || 0;
            digit = val1 + val2 + carry;
            if (digit < 10) {
                carry = 0;
            }
            else {
                carry = 1;
                digit -= 10;
            }
            result.unshift(digit);
        }
        if (carry) {
            result.unshift(1);
        }
        return new GiantInteger(result);
    }
    g_increment(g) {
        this.q = this.g_sum(g).q;
    }
    g_product(n) {
        let result = this.g_copy();
        for (let i = 0; i < n - 1; ++i) {
            result.g_increment(this);
        }
        return result;
    }
    stringify() {
        if (!this.q.length) {
            return "0";
        }
        return this.q.join("");
    }
}
class Core {
    constructor(shape, atom, view_dim) {
        this.shape = shape;
        this.atom = atom;
        this.view_dim = view_dim;
        this.dim = shape.length;
        this.strides = this.get_strides();
        this.reversed_strides = copy(this.strides);
        this.reversed_strides.reverse();
        this.size = prod(shape);
        this.grid = new Array(this.size).fill(0);
        this.previous_grid = new Array(this.size).fill(0);
        this.total_score = new GiantInteger();
        this.previous_total_score = new GiantInteger();
        this.game_over = false;
        this.powers = [];
        for (let power = 0; power <= 2; ++power) {
            this.powers.push(new GiantInteger(Math.pow(atom, power)));
        }
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
    push_power() {
        this.powers.push(this.powers[this.powers.length - 1].g_product(this.atom));
    }
    simulate_move(axis, opposite) {
        let grid = this.grid;
        let ngrid = new Array(this.size).fill(0);
        let link_data = new Map();
        let score = new GiantInteger();
        let abs_minor_step = this.strides[axis];
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
        let val;
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
                    val = grid[index];
                    if (val === 0) {
                        continue;
                    }
                    else if (val === hold) {
                        ++count;
                        temp_indices.push(index);
                        if (count === this.atom) {
                            ngrid[ngrid_p] = val + 1;
                            link_data.set(ngrid_p, temp_indices);
                            while (this.powers.length <= val + 1) {
                                this.push_power();
                            }
                            score.g_increment(this.powers[val + 1]);
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
                        hold = val;
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
            this.previous_total_score = this.total_score.g_copy();
            this.total_score.g_increment(move_tiles_result.score);
            // There must be at least one available cell to insert a new tile
            let new_tile_index = this.generate_tile_randomly();
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
        for (let val of this.grid) {
            if (!val) {
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
        return !this.has_available_cells() && this.no_matches_available();
    }
    get_available_cell_indices() {
        let result = [];
        this.grid.forEach((val, index) => {
            if (!val) {
                result.push(index);
            }
        });
        return result;
    }
    generate_random_tile_val() {
        return Math.random() < 0.9 ? 1 : 2;
    }
    generate_tile_randomly() {
        let available_cell_indices = this.get_available_cell_indices();
        if (!available_cell_indices.length) {
            return -1;
        }
        let index = sample(available_cell_indices);
        let val = this.generate_random_tile_val();
        this.grid[index] = val;
        return index;
    }
    snake_fill_grid() {
        // This works only when atom === 2
        let axis;
        let indices = new Array(this.dim).fill(0);
        let steps = new Array(this.dim).fill(1);
        for (let val = 0; val < this.size - 1; ++val) {
            this.grid[this.ravel(indices)] = val;
            axis = 0;
            indices[axis] += steps[axis];
            while (indices[axis] === -1 || indices[axis] === this.shape[axis]) {
                if (indices[axis] === -1) {
                    ++indices[axis];
                    steps[axis] = 1;
                }
                else {
                    --indices[axis];
                    steps[axis] = -1;
                }
                ++axis;
                indices[axis] += steps[axis];
            }
        }
        this.grid[this.ravel(indices)] = this.size - 1;
        this.grid[0] = 1;
    }
    serialize() {
        return {
            shape: this.shape,
            atom: this.atom,
            view_dim: this.view_dim,
            grid: this.grid,
            total_score: this.total_score,
            game_over: this.game_over,
        };
    }
}
class Recorder {
    constructor() {
        this.game_state_key = "game_state";
        this.storage = window.localStorage;
    }
    get_game_state() {
        let state_json = this.storage.getItem(this.game_state_key);
        return state_json ? JSON.parse(state_json) : null;
    }
    set_game_state(game_state) {
        this.storage.setItem(this.game_state_key, JSON.stringify(game_state));
    }
    clear_game_state() {
        this.storage.removeItem(this.game_state_key);
    }
}
class Game extends Core {
    constructor(new_game_info = null) {
        let recorder = new Recorder();
        let is_new_game = true;
        if (new_game_info) {
            super(new_game_info.shape, new_game_info.atom, new_game_info.view_dim);
        }
        else {
            let recorded = recorder.get_game_state();
            if (recorded) {
                is_new_game = false;
                super(recorded.shape, recorded.atom, recorded.view_dim);
                this.grid = recorded.grid;
                this.previous_grid = this.grid;
                this.total_score = new GiantInteger(recorded.total_score.q);
                this.previous_total_score = this.total_score.g_copy();
                this.game_over = recorded.game_over;
                this.refresh_grid();
            }
            else {
                super([4, 4], 2, 2);
            }
        }
        this.c = {
            tile_style_limit: 14,
            begin_tiles: 2,
            page_coverage: 0.8,
            lowest_level_buff: 0.15,
            buff_gap: 0.3,
            max_text_length_upper_limit: 9,
            tile_size_intervals_px: [35, 50, 70, 90, 100],
            // To be computed
            frame_width: 0,
            full_width: 0,
            full_height: 0,
            tile_size_px: 0,
            game_field_width_px: 0,
            game_field_height_px: 0,
            max_text_length: 0,
        };
        this.recorder = recorder;
        this.init(is_new_game);
    }
    init(is_new_game) {
        this.init_consts();
        this.init_client_related_consts();
        this.init_positions();
        this.init_client_related_sizes();
        if (is_new_game) {
            this.add_start_tiles();
            this.update_score();
        }
        else if (this.total_score) {
            window.requestAnimationFrame(() => {
                this.update_score_animation(this.total_score);
            });
        }
    }
    get_cell_anchor_point(index) {
        let result = [];
        let indices = this.unravel(index);
        let view_axis;
        let current_unit;
        let axis_val;
        let axis;
        for (view_axis = 0; view_axis < this.view_dim; ++view_axis) {
            current_unit = 1.0 + this.c.lowest_level_buff;
            axis_val = this.c.frame_width;
            for (axis = view_axis; axis < this.dim; axis += this.view_dim) {
                axis_val += indices[axis] * current_unit;
                current_unit *= this.shape[axis];
                current_unit += this.c.buff_gap;
            }
            result.push(axis_val);
        }
        return result;
    }
    get_full_grid_size() {
        let result = [];
        let added_width = 1.0 + this.c.frame_width;
        this.get_cell_anchor_point(this.size - 1).forEach((axis_val) => {
            result.push(axis_val + added_width);
        });
        return result;
    }
    css(id_name, rule_command) {
        // Can this be the best practice?
        let style = query_selector(`#${id_name}`);
        if (style) {
            style.textContent = rule_command;
        }
        else {
            style = document.createElement("style");
            style.id = id_name;
            style.textContent = rule_command;
            document.head.appendChild(style);
        }
    }
    get_2D_translate_string(translate_list) {
        let translate_str_list = [];
        translate_list.forEach((ratio) => {
            translate_str_list.push(`${ratio}em`);
        });
        let translate_str = translate_str_list.join(", ");
        return `translate(${translate_str})`;
    }
    get_text_string(val) {
        let power = Math.pow(this.atom, val);
        if (power >= Math.pow(10, this.c.max_text_length)) {
            let logarithm = Math.log10(this.atom) * val;
            let exponent = Math.floor(logarithm);
            if (exponent >= 100) {
                let power_string = `${this.atom}^${val}`;
                if (power_string.length > this.c.max_text_length) {
                    return "INF";
                }
                else {
                    return `${this.atom}^${val}`;
                }
            }
            let decimals = this.c.max_text_length - String(exponent).length - 3;
            let coefficient = Math.round(Math.pow(10, logarithm - exponent + decimals));
            let coefficient_str = String(coefficient);
            return `${coefficient_str[0]}.${coefficient_str.slice(1)}E${exponent}`;
        }
        else {
            return String(power);
        }
    }
    init_consts() {
        this.c.frame_width = Math.ceil(this.dim / this.view_dim - 1) * this.c.buff_gap + this.c.lowest_level_buff;
        let full_grid_size = this.get_full_grid_size();
        this.c.full_width = full_grid_size[0];
        this.c.full_height = full_grid_size[1];
    }
    init_client_related_consts() {
        let intervals = this.c.tile_size_intervals_px;
        let client_width = document.body.clientWidth * this.c.page_coverage;
        let tile_size_px = client_width / this.c.full_width;
        tile_size_px = Math.max(tile_size_px, intervals[0]);
        tile_size_px = Math.min(tile_size_px, intervals[intervals.length - 1]);
        if (this.c.tile_size_px === tile_size_px) {
            return false; // No variables changed
        }
        this.c.tile_size_px = tile_size_px;
        this.c.game_field_width_px = tile_size_px * this.c.full_width;
        this.c.game_field_height_px = tile_size_px * this.c.full_height;
        let max_text_length = this.c.max_text_length_upper_limit;
        intervals.slice(1, intervals.length - 1).forEach((interval) => {
            if (tile_size_px < interval) {
                --max_text_length;
            }
        });
        this.c.max_text_length = max_text_length;
        return true;
    }
    init_positions() {
        this.css("game_container_size", `.game_container {
            height: ${this.c.full_height}em;
            border-radius: ${this.c.frame_width}em;
        }`); // Width is inherited
        let anchor_point;
        let transform_str;
        let cell;
        let tile_position_commands = [];
        for (let index = 0; index < this.size; ++index) {
            anchor_point = this.get_cell_anchor_point(index);
            transform_str = this.get_2D_translate_string(anchor_point);
            tile_position_commands.push(`.tile_position_${index} {transform: ${transform_str};}`);
            cell = document.createElement("div");
            cell.classList.add("grid_cell", `tile_position_${index}`);
            $(".grid_container").appendChild(cell);
        }
        this.css("tile_positions", tile_position_commands.join("\n"));
    }
    init_client_related_sizes() {
        this.css("container_size", `.container {
            font-size: ${this.c.tile_size_px}px;
            width: ${this.c.game_field_width_px}px;
        }`);
    }
    add_start_tiles() {
        let index;
        for (let i = 0; i < this.c.begin_tiles; ++i) {
            index = this.generate_tile_randomly();
            this.add_tile(index);
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
    add_tile(index, previous = null, val = 0) {
        if (!val) {
            val = this.grid[index];
        }
        let inner_text = this.get_text_string(val);
        let tile = document.createElement("div");
        let classes = [
            "tile",
            `tile_position_${index}`,
            `tile_val_${Math.min(val, this.c.tile_style_limit)}`,
            `tile_text_length_${inner_text.length}`,
        ];
        this.apply_classes(tile, classes);
        if (typeof previous === "number") {
            // Make sure that the tile gets rendered in the previous position first
            let previous_classes = copy(classes);
            previous_classes[1] = `tile_position_${previous}`;
            this.apply_classes(tile, previous_classes);
            window.requestAnimationFrame(() => {
                this.apply_classes(tile, classes);
            });
        }
        else if (previous) {
            classes.push("tile_merged");
            this.apply_classes(tile, classes);
            // Render the tiles that merged
            previous.forEach((merged_index) => {
                this.add_tile(index, merged_index, val - 1);
            });
        }
        else {
            classes.push("tile_new");
            this.apply_classes(tile, classes);
        }
        let tile_inner = document.createElement("div");
        tile_inner.classList.add("tile_inner");
        let tile_text = document.createElement("div");
        tile_text.classList.add("tile_text");
        tile_text.textContent = inner_text;
        tile_inner.appendChild(tile_text);
        tile.appendChild(tile_inner);
        $(".tile_container").appendChild(tile);
    }
    clear_tiles() {
        this.clear_container($(".tile_container"));
    }
    update_score() {
        $(".score_value").textContent = this.total_score.stringify();
    }
    update_score_animation(score) {
        this.clear_container($(".score_value"));
        this.update_score();
        if (score.q.length) {
            let addition = document.createElement("div");
            addition.classList.add("score_addition");
            addition.textContent = `+${score.stringify()}`;
            $(".score_value").appendChild(addition);
        }
    }
    show_game_over() {
        this.game_over = true;
        $(".game_message").classList.add("game_over_message");
        $(".game_message").textContent = "Game over!";
    }
    clear_game_over() {
        this.game_over = false;
        $(".game_message").classList.remove("game_over_message");
    }
    actuate(axis, opposite) {
        let move_tiles_result = this.move_tiles(axis, opposite);
        if (!move_tiles_result.changed) {
            return false;
        }
        window.requestAnimationFrame(() => {
            this.clear_tiles();
            move_tiles_result.link_data.forEach((previous, index) => {
                this.add_tile(index, previous);
            });
            let score = move_tiles_result.score;
            this.update_score_animation(score);
            if (move_tiles_result.game_over) {
                this.show_game_over();
                this.delete_saving();
            }
            else {
                this.save_game();
            }
        });
        return true;
    }
    refresh_grid() {
        window.requestAnimationFrame(() => {
            this.clear_tiles();
            this.grid.forEach((val, index) => {
                if (val) {
                    this.add_tile(index);
                }
            });
            this.update_score();
        });
    }
    new_game() {
        this.delete_saving();
        for (let index = 0; index < this.size; ++index) {
            this.grid[index] = 0;
        }
        this.total_score = new GiantInteger();
        this.previous_total_score = new GiantInteger();
        this.clear_game_over();
        this.add_start_tiles();
        this.refresh_grid();
    }
    withdraw() {
        this.grid = this.previous_grid;
        this.total_score = this.previous_total_score.g_copy();
        this.clear_game_over();
        this.save_game();
        this.refresh_grid();
    }
    save_game() {
        this.recorder.set_game_state(this.serialize());
    }
    delete_saving() {
        this.recorder.clear_game_state();
    }
    keyboard_event(event_key) {
        if (this.game_over) {
            return;
        }
        let choice = "lkhgdsjiftaw".indexOf(event_key);
        let axis = choice % 6;
        let opposite = choice < 6;
        if (choice < 0 || axis >= this.dim) {
            return;
        }
        this.actuate(axis, opposite);
    }
    window_onresize() {
        let changed = this.init_client_related_consts();
        if (changed) {
            this.init_client_related_sizes();
            this.refresh_grid();
        }
    }
    snake_fill_tiles() {
        this.snake_fill_grid();
        this.refresh_grid();
    }
}
var game = new Game();
function new_size_game(shape, atom, view_dim) {
    game.clear_container($(".grid_container"));
    game.clear_tiles();
    game.clear_game_over();
    game.delete_saving();
    game = new Game({ shape: shape, atom: atom, view_dim: view_dim });
}
function check_validity(test_string, valid_chars) {
    let result_string = test_string.replace(/\s*/g, ""); // Ignore spaces
    for (let char of result_string) {
        if (valid_chars.indexOf(char) === -1) {
            return "";
        }
    }
    return result_string;
}
document.addEventListener("keypress", (event) => {
    event.preventDefault();
    game.keyboard_event(event.key);
});
window.onresize = function () {
    game.window_onresize();
};
$(".withdraw_button").addEventListener("click", () => {
    game.withdraw();
});
$(".auto_play_button").addEventListener("click", () => {
    alert("To be implemented...");
});
$(".auto_play_speed_button").addEventListener("click", () => {
    alert("To be implemented...");
});
$(".new_game_button").addEventListener("click", () => {
    game.new_game();
});
$(".new_size_button").addEventListener("click", () => {
    let invalid_input_message = "Invalid input!";
    let shape_input = prompt("Type in the dimensions separated with commas\n"
        + "(Eeach value should be at least 2):", "4, 4, 2");
    if (!shape_input) {
        return;
    }
    shape_input = check_validity(shape_input, "0123456789,");
    if (!shape_input.length) {
        alert(invalid_input_message);
        return;
    }
    let separated_shape_input = shape_input.split(",");
    let shape = [];
    let dimension_size;
    for (let dimension_size_input of separated_shape_input) {
        dimension_size = Number(dimension_size_input);
        if (dimension_size < 2) {
            alert(invalid_input_message);
            return;
        }
        shape.push(dimension_size);
    }
    let atom_input = prompt("Type in the number of tiles to merge (at least 2):", "2");
    if (!atom_input) {
        return;
    }
    atom_input = check_validity(atom_input, "0123456789");
    if (!atom_input.length) {
        alert(invalid_input_message);
        return;
    }
    let atom = Number(atom_input);
    if (atom < 2) {
        alert(invalid_input_message);
        return;
    }
    new_size_game(shape, atom, 2);
});
$(".key_bindings_button").addEventListener("click", () => {
    alert("To be implemented...");
});
// TODO: clear all typescript compiling warnings
// use css variables
// css style -> var(...) (inside scss)
// curtain
// text paragraph
// classes
// disable withdraw button
// modify README
