"use strict";
function copy(array) {
    return [...array];
}
function prod(array) {
    let result = 1;
    array.forEach((val) => {
        result *= val;
    });
    return result;
}
function randint(upper_bound) {
    return Math.floor(Math.random() * upper_bound);
}
function sample(samples) {
    return samples[randint(samples.length)];
}
function array_equal(arr0, arr1) {
    return arr0.toString() === arr1.toString();
}
function is_unique_str(str) {
    let str_copy = str;
    while (str_copy.length) {
        if (str_copy.slice(1).indexOf(str_copy.charAt(0)) !== -1) {
            return false;
        }
        str_copy = str_copy.slice(1);
    }
    return true;
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
    renew(shape, atom, view_dim) {
        if (!array_equal(shape, this.shape)) {
            this.shape = shape;
            this.dim = shape.length;
            this.strides = this.get_strides();
            this.reversed_strides = copy(this.strides);
            this.reversed_strides.reverse();
            this.size = prod(shape);
        }
        if (atom !== this.atom) {
            this.atom = atom;
            this.powers = [];
            for (let power = 0; power <= 2; ++power) {
                this.powers.push(new GiantInteger(Math.pow(atom, power)));
            }
        }
        this.view_dim = view_dim;
        this.renew_self();
    }
    renew_self() {
        this.grid = new Array(this.size).fill(0);
        this.previous_grid = new Array(this.size).fill(0);
        this.total_score = new GiantInteger();
        this.previous_total_score = new GiantInteger();
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
            for (let opposite of [true, false]) {
                if (this.simulate_move(axis, opposite).changed) {
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
}
class Recorder {
    constructor() {
        this.game_state_key = "game_state";
        this.tile_size_key = "tile_size";
        this.key_bindings_key = "key_bindings";
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
    get_tile_size() {
        let state_json = this.storage.getItem(this.tile_size_key);
        return state_json ? Number(state_json) : null;
    }
    set_tile_size(tile_size_px) {
        this.storage.setItem(this.tile_size_key, String(tile_size_px));
    }
    clear_tile_size() {
        this.storage.removeItem(this.tile_size_key);
    }
    get_key_bindings() {
        return this.storage.getItem(this.key_bindings_key);
    }
    set_key_bindings(key_bindings) {
        this.storage.setItem(this.key_bindings_key, key_bindings);
    }
    clear_key_bindings() {
        this.storage.removeItem(this.key_bindings_key);
    }
}
class Game extends Core {
    constructor() {
        let c = {
            tile_style_limit: 14,
            begin_tiles: 2,
            page_coverage: 0.8,
            lowest_level_buff: 0.15,
            buff_gap: 0.3,
            default_tile_size_px: 100,
            tile_size_intervals: [
                35, 40, 50, 60, 70, 80, 90, 100,
                120, 140, 160, 180, 200, 225, 250
            ],
            tile_size_interpolations: [/*6*/ 50 /*7*/, 70 /*8*/, 90 /*9*/],
            max_text_length_upper_limit: 9,
            default_key_bindings: "jlikfhtgadws",
            default_auto_move_time: 1000,
            // To be computed
            frame_width: 0,
            full_width: 0,
            full_height: 0,
            max_text_length: 0,
        };
        let recorder = new Recorder();
        let is_new_game;
        let recorded = recorder.get_game_state();
        if (recorded) {
            is_new_game = false;
            super(recorded.shape, recorded.atom, recorded.view_dim);
            this.grid = recorded.grid;
            this.previous_grid = this.grid;
            this.total_score = new GiantInteger(recorded.total_score.q);
            this.previous_total_score = this.total_score.g_copy();
            this.game_over = recorded.game_over;
        }
        else {
            is_new_game = true;
            super([4, 4], 2, 2);
        }
        this.tile_size_px = recorder.get_tile_size() || c.default_tile_size_px;
        this.key_bindings = recorder.get_key_bindings()
            || c.default_key_bindings.slice(0, 2 * this.dim);
        this.auto_move_time = c.default_auto_move_time;
        this.auto_move_flag = false;
        this.c = c;
        this.recorder = recorder;
        this.init(is_new_game);
    }
    init(is_new_game) {
        $(".curtain").hide();
        this.refresh_grid();
        this.init_consts();
        this.init_positions();
        this.init_font_size();
        this.init_max_text_length();
        if (is_new_game) {
            this.add_start_tiles();
            this.update_score();
        }
        else if (this.total_score) {
            window.requestAnimationFrame(() => {
                this.update_score_animation(this.total_score);
            });
        }
        this.disable_btn(".withdraw_button");
        this.toggle_zoom_button();
        $(".text_atom").text(this.atom);
    }
    disable_btn(button_selector) {
        $(button_selector).attr("disabled", true);
    }
    active_btn(button_selector) {
        $(button_selector).attr("disabled", false);
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
    get_2D_translate_string(translate_list) {
        let translate_str_list = [];
        translate_list.forEach((ratio) => {
            translate_str_list.push(`${ratio}em`);
        });
        let translate_str = translate_str_list.join(", ");
        return `translate(${translate_str})`;
    }
    get_tile_transform_string(index) {
        return this.get_2D_translate_string(this.get_cell_anchor_point(index));
    }
    get_text_string(val) {
        let max_text_length = this.c.max_text_length;
        let power = Math.pow(this.atom, val);
        if (power >= Math.pow(10, max_text_length)) {
            let logarithm = Math.log10(this.atom) * val;
            let exponent = Math.floor(logarithm);
            if (exponent >= 100) {
                let power_string = `${this.atom}^${val}`;
                if (power_string.length > max_text_length) {
                    return "INF";
                }
                else {
                    return `${this.atom}^${val}`;
                }
            }
            let decimals = max_text_length - String(exponent).length - 3;
            let coefficient = String(Math.round(Math.pow(10, logarithm - exponent + decimals)));
            return `${coefficient[0]}.${coefficient.slice(1)}E${exponent}`;
        }
        else {
            return String(power);
        }
    }
    init_consts() {
        this.c.frame_width = Math.ceil(this.dim / this.view_dim - 1)
            * this.c.buff_gap + this.c.lowest_level_buff;
        let full_grid_size = this.get_full_grid_size();
        this.c.full_width = full_grid_size[0];
        this.c.full_height = full_grid_size[1];
    }
    init_max_text_length() {
        let max_text_length = this.c.max_text_length_upper_limit;
        this.c.tile_size_interpolations.forEach((interval) => {
            if (this.tile_size_px < interval) {
                --max_text_length;
            }
        });
        this.c.max_text_length = max_text_length;
    }
    init_positions() {
        $(".game_container")
            .css("width", `${this.c.full_width}em`)
            .css("height", `${this.c.full_height}em`)
            .css("border-radius", `${this.c.frame_width}em`);
        for (let index = 0; index < this.size; ++index) {
            $("<div></div>").addClass("grid_cell")
                .css("transform", this.get_tile_transform_string(index))
                .appendTo(".grid_container");
        }
    }
    init_font_size() {
        $(".game_container").css("font-size", `${this.tile_size_px}px`);
    }
    add_start_tiles() {
        let index;
        for (let i = 0; i < this.c.begin_tiles; ++i) {
            index = this.generate_tile_randomly();
            this.add_tile(index);
        }
        this.previous_grid = this.grid;
    }
    add_tile(index, previous = null, val = 0) {
        if (!val) {
            val = this.grid[index];
        }
        let inner_text = this.get_text_string(val);
        let tile = $("<div></div>").addClass([
            "tile",
            `tile_val_${Math.min(val, this.c.tile_style_limit)}`,
            `tile_text_length_${inner_text.length}`
        ].join(" "));
        tile.css("transform", this.get_tile_transform_string(index));
        if (typeof previous === "number") {
            // Make sure that the tile gets rendered
            // in the previous position first
            tile.css("transform", this.get_tile_transform_string(previous));
            window.requestAnimationFrame(() => {
                tile.css("transform", this.get_tile_transform_string(index));
            });
        }
        else if (previous) {
            tile.addClass("tile_merged");
            // Render the tiles that merged
            previous.forEach((merged_index) => {
                this.add_tile(index, merged_index, val - 1);
            });
        }
        else {
            tile.addClass("tile_new");
        }
        $(".tile_container").append(tile.append($("<div></div>").addClass("tile_inner").append($("<div></div>").addClass("tile_text").text(inner_text))));
    }
    clear_tiles() {
        $(".tile_container").empty();
    }
    update_score() {
        $(".score_value").text(this.total_score.stringify());
    }
    update_score_animation(score) {
        $(".score_value").empty();
        this.update_score();
        if (score.q.length) {
            $("<div></div>").addClass("score_addition")
                .text(`+${score.stringify()}`)
                .appendTo(".score_value");
        }
    }
    show_game_over() {
        this.game_over = true;
        $(".game_message").addClass("game_over_message").text("Game over!");
    }
    clear_game_over() {
        this.game_over = false;
        $(".game_message").removeClass("game_over_message");
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
            this.active_btn(".withdraw_button");
            if (move_tiles_result.game_over) {
                this.show_game_over();
                if (this.auto_move_flag) {
                    this.toggle_auto_move();
                }
                this.disable_btn(".random_play_button");
                this.disable_btn(".random_play_speed_button");
                this.delete_game_saving();
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
    retreat() {
        if (this.auto_move_flag) {
            this.toggle_auto_move();
        }
        this.clear_game_over();
        this.refresh_grid();
        this.disable_btn(".withdraw_button");
        this.active_btn(".random_play_button");
        this.active_btn(".random_play_speed_button");
    }
    withdraw() {
        this.grid = this.previous_grid;
        this.total_score = this.previous_total_score.g_copy();
        this.save_game();
        this.retreat();
    }
    new_game() {
        this.delete_game_saving();
        this.renew_self();
        this.add_start_tiles();
        this.retreat();
    }
    new_size_game(shape, atom, view_dim) {
        $(".grid_container").empty();
        this.clear_tiles();
        this.delete_game_saving();
        this.renew(shape, atom, view_dim);
        this.init(true);
        this.key_bindings = this.c.default_key_bindings.slice(0, 2 * this.dim);
        this.save_key_bindings();
        this.retreat();
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
    save_game() {
        this.recorder.set_game_state(this.serialize());
    }
    delete_game_saving() {
        this.recorder.clear_game_state();
    }
    save_tile_size() {
        this.recorder.set_tile_size(this.tile_size_px);
    }
    save_key_bindings() {
        this.recorder.set_key_bindings(this.key_bindings);
    }
    keyboard_event(event_key) {
        if (this.game_over) {
            return;
        }
        let direction = this.key_bindings.indexOf(event_key);
        if (direction === -1) {
            return;
        }
        let axis = Math.floor(direction / 2);
        let opposite = Boolean(direction % 2);
        this.actuate(axis, opposite);
    }
    modify_key_bindings(new_key_bindings) {
        this.key_bindings = new_key_bindings;
        this.save_key_bindings();
    }
    auto_move() {
        if (!this.auto_move_flag) {
            return;
        }
        let axis = randint(this.dim);
        let opposite = Boolean(randint(2));
        if (!this.simulate_move(axis, opposite).changed) {
            this.auto_move();
        }
        else {
            this.actuate(axis, opposite);
            setTimeout("game.auto_move()", this.auto_move_time);
        }
    }
    toggle_auto_move() {
        if (!this.auto_move_flag) {
            this.auto_move_flag = true;
            $(".random_play_button").text("Stop");
            this.auto_move();
        }
        else {
            this.auto_move_flag = false;
            $(".random_play_button").text("Random Play");
        }
    }
    modify_auto_move_time(new_auto_move_time) {
        this.auto_move_time = new_auto_move_time;
        if (!this.auto_move_flag) {
            this.toggle_auto_move();
        }
    }
    modify_tile_size_px(tile_size_px) {
        this.tile_size_px = tile_size_px;
        this.init_font_size();
        this.init_max_text_length();
        this.save_tile_size();
        this.refresh_grid();
    }
    toggle_zoom_button() {
        if (this.tile_size_px === this.c.tile_size_intervals.slice(-1)[0]) {
            this.disable_btn(".zoom-in_button");
        }
        else {
            this.active_btn(".zoom-in_button");
        }
        if (this.tile_size_px === this.c.tile_size_intervals[0]) {
            this.disable_btn(".zoom-out_button");
        }
        else {
            this.active_btn(".zoom-out_button");
        }
    }
    zoom_in() {
        this.modify_tile_size_px(this.c.tile_size_intervals[this.c.tile_size_intervals.indexOf(this.tile_size_px) + 1]);
        this.toggle_zoom_button();
    }
    zoom_out() {
        this.modify_tile_size_px(this.c.tile_size_intervals[this.c.tile_size_intervals.indexOf(this.tile_size_px) - 1]);
        this.toggle_zoom_button();
    }
    snake_fill_tiles() {
        this.snake_fill_grid();
        this.refresh_grid();
    }
}
var game;
// Wait till the browser is ready to render the game (avoids glitches)
window.requestAnimationFrame(() => {
    game = new Game();
});
document.addEventListener("keypress", (event) => {
    event.preventDefault();
    game.keyboard_event(event.key);
});
$(".withdraw_button").click(() => {
    game.withdraw();
});
$(".random_play_button").click(() => {
    game.toggle_auto_move();
});
$(".random_play_speed_button").click(() => {
    let invalid_input_message = "Invalid input!";
    let auto_move_time_input = prompt("Type in the speed of random playing\n" +
        "(unit: ms / move):", String(game.auto_move_time));
    if (!auto_move_time_input) {
        return;
    }
    if (!/^(\d*|\d*\.\d*)$/.test(auto_move_time_input)) {
        alert(invalid_input_message);
        return;
    }
    let auto_move_time = Number(auto_move_time_input);
    game.modify_auto_move_time(auto_move_time);
});
$(".new_game_button").click(() => {
    game.new_game();
});
$(".new_size_button").click(() => {
    let invalid_input_message = "Invalid input!";
    let shape_input = prompt("Type in the dimensions separated with commas\n" +
        "(each value should be at least 2):", "4, 4, 2");
    if (!shape_input) {
        return;
    }
    shape_input = shape_input.replace(/\s*/g, "");
    if (!/^[\d,]*$/.test(shape_input)) {
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
    let biggest_dim_size = Math.max(...shape);
    let atom;
    if (biggest_dim_size === 2) {
        atom = 2;
    }
    else {
        let atom_input = prompt("Type in the number of tiles to merge\n" +
            `(at least 2, at most ${biggest_dim_size}):`, "2");
        if (!atom_input) {
            return;
        }
        if (!/^\d*$/.test(atom_input)) {
            alert(invalid_input_message);
            return;
        }
        atom = Number(atom_input);
        if (atom < 2 || atom > biggest_dim_size) {
            alert(invalid_input_message);
            return;
        }
    }
    game.new_size_game(shape, atom, 2);
});
$(".key_bindings_button").click(() => {
    let invalid_input_message = "Invalid input!";
    let padded_key_bindings = game.key_bindings.padEnd(game.dim * 2, "?");
    let original_character_pairs = [];
    for (let axis = 0; axis < game.dim; ++axis) {
        original_character_pairs.push(padded_key_bindings.substr(2 * axis, 2));
    }
    let key_bindings_input = prompt("Each pair of lowercase characters separated with commas\n" +
        "controls a pair of opposite directions of a specified axis.\n" +
        "Make sure all the characters are unique,\n" +
        "and the number of pairs fits the number of axes:", original_character_pairs.join(", "));
    if (!key_bindings_input) {
        return;
    }
    key_bindings_input = key_bindings_input.replace(/\s*/g, "");
    if (!/^[a-z,]*$/.test(key_bindings_input)) {
        alert(invalid_input_message);
        return;
    }
    let separated_key_bindings_input = key_bindings_input.split(",");
    if (separated_key_bindings_input.length !== game.dim) {
        alert(invalid_input_message);
        return;
    }
    for (let character_pair of separated_key_bindings_input) {
        if (character_pair.length !== 2) {
            alert(invalid_input_message);
            return;
        }
    }
    let key_bindings = separated_key_bindings_input.join("");
    if (!is_unique_str(key_bindings)) {
        alert(invalid_input_message);
        return;
    }
    game.modify_key_bindings(key_bindings);
});
$(".zoom-in_button").click(() => {
    game.zoom_in();
});
$(".zoom-out_button").click(() => {
    game.zoom_out();
});
// TODO: clear all typescript compiling warnings
// classes
// fancy font
// mobile friendly
