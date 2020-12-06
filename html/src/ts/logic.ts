function copy<T>(array: T[]): T[] {
    let result: T[] = [];
    array.forEach((val: T) => {
        result.push(val);
    });
    return result;
}


function prod(array: number[]): number {
    let result: number = 1;
    array.forEach((val: number) => {
        result *= val;
    })
    return result;
}


function sample(samples: number[]): number {
    return samples[Math.floor(Math.random() * samples.length)];
}


type PreviousType = number | number[] | null;


interface NewGameInfo {
    shape: number[],
    atom: number,
    view_dim: number,
}


interface GameState {
    shape: number[],
    atom: number,
    view_dim: number,
    grid: number[],
    total_score: GiantNumber,
    game_over: boolean,
}


interface MoveTilesResult {
    link_data: Map<number, PreviousType>,
    new_grid: number[],
    changed: boolean,
    score: GiantNumber,
    game_over: boolean,
}


interface Constants {
    tile_style_limit: number,
    begin_tiles: number,
    min_max_text_length: number,
    page_coverage: number,
    lowest_level_buff: number,
    buff_gap: number,
    tile_size_intervals_px: number[],
    frame_width: number,
    full_width: number,
    full_height: number,
    tile_size_px: number,
    game_field_width_px: number,
    game_field_height_px: number,
    max_text_length: number,
}


class GiantNumber {
    v: number[];

    constructor(val: number[] | number = 0) {
        if (typeof val === "number") {
            if (!val) {
                this.v = [0];
            } else {
                this.v = [];
                let num: number = val;
                while (num) {
                    this.v.unshift(num % 10);
                    num = Math.floor(num / 10);
                }
            }
        } else {
            this.v = val;
        }
    }

    g_copy(): GiantNumber {
        return new GiantNumber(this.v);
    }

    g_sum(g: GiantNumber): GiantNumber {
        if (this.v === [0]) {
            return g;
        }
        if (g.v === [0]) {
            return this;
        }
        let result: number[] = [];
        let carry: number = 0;
        let l1: number[] = copy<number>(this.v);
        let l2: number[] = copy<number>(g.v);
        let val1: number;
        let val2: number;
        let digit: number;
        while (l1.length || l2.length) {
            val1 = l1.pop() || 0;
            val2 = l2.pop() || 0;
            digit = val1 + val2 + carry;
            if (digit < 10) {
                carry = 0;
            } else {
                carry = 1;
                digit -= 10;
            }
            result.unshift(digit);
        }
        if (carry) {
            result.unshift(1);
        }
        return new GiantNumber(result);
    }

    g_increment(g: GiantNumber): void {
        this.v = this.g_sum(g).v;
    }

    g_product(n: number): GiantNumber {
        let result: GiantNumber = this.g_copy();
        for (let i: number = 0; i < n - 1; ++i) {
            result.g_increment(this);
        }
        return result;
    }

    stringify(): string {
        return this.v.join("");
    }
}


class Core {
    shape: number[];
    atom: number;
    view_dim: number;
    dim: number;
    strides: number[];
    reversed_strides: number[];
    size: number;
    grid: number[];
    previous_grid: number[];
    powers: GiantNumber[];
    total_score: GiantNumber;
    previous_total_score: GiantNumber;
    game_over: boolean;

    constructor(shape: number[], atom: number, view_dim: number) {
        this.shape = shape;
        this.atom = atom;
        this.view_dim = view_dim;
        this.dim = shape.length;
        this.strides = this.get_strides();
        this.reversed_strides = copy<number>(this.strides);
        this.reversed_strides.reverse();
        this.size = prod(shape);
        this.grid = new Array(this.size).fill(0);
        this.previous_grid = new Array(this.size).fill(0);
        this.total_score = new GiantNumber();
        this.previous_total_score = new GiantNumber();
        this.game_over = false;
        this.powers = [];
        for (let power = 0; power <= 2; ++power) {
            this.powers.push(new GiantNumber(Math.pow(atom, power)));
        }
    }

    get_strides(): number[] {
        let result: number[] = [];
        let temp_stride: number = 1;
        this.shape.forEach((dimension_size: number) => {
            result.push(temp_stride);
            temp_stride *= dimension_size;
        });
        return result;
    }

    ravel(indices: number[]): number {
        let result: number = 0;
        this.strides.forEach((stride: number, axis: number) => {
            result += indices[axis] * stride;
        });
        return result;
    }

    unravel(index: number): number[] {
        let result: number[] = [];
        this.reversed_strides.forEach((stride: number) => {
            result.unshift(Math.floor(index / stride));
            index = index % stride;
        });
        return result;
    }

    push_power() {
        this.powers.push(this.powers[this.powers.length - 1].g_product(this.atom));
    }

    simulate_move(axis: number, opposite: boolean): MoveTilesResult {
        let grid: number[] = this.grid;
        let ngrid: number[] = new Array(this.size).fill(0);
        let link_data: Map<number, PreviousType> = new Map<number, PreviousType>();
        let score: GiantNumber = new GiantNumber();
        let abs_minor_step: number = this.strides[axis];
        let minor_step: number = opposite ? -abs_minor_step : abs_minor_step;
        let major_step: number = abs_minor_step * this.shape[axis];
        let i0: number;
        let i1: number;
        let begin_index: number;
        let ngrid_p: number;
        let hold: number;
        let count: number;
        let i2: number;
        let index: number;
        let val: number;
        let exact_i3: number;
        let temp_indices: number[];
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
                    } else if (val === hold) {
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
                    } else {
                        temp_indices.forEach((temp_indice: number, i3: number) => {
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
                temp_indices.forEach((temp_indice: number, i3: number) => {
                    exact_i3 = ngrid_p + i3 * minor_step;
                    ngrid[exact_i3] = hold;
                    link_data.set(exact_i3, temp_indice);
                });
            }
        }
        let changed: boolean = !this.all_equal(ngrid);

        return {
            link_data: link_data,
            new_grid: ngrid,
            changed: changed,
            score: score,
            game_over: false
        };
    }

    move_tiles(axis: number, opposite: boolean): MoveTilesResult {
        let move_tiles_result: MoveTilesResult = this.simulate_move(axis, opposite);
        if (move_tiles_result.changed) {
            this.previous_grid = this.grid;
            this.grid = move_tiles_result.new_grid;
            this.previous_total_score = this.total_score.g_copy();
            this.total_score.g_increment(move_tiles_result.score);

            // There must be at least one available cell to insert a new tile
            let new_tile_index: number = this.generate_tile_randomly();
            move_tiles_result.link_data.set(new_tile_index, null);

            move_tiles_result.game_over = this.check_if_game_over();
        }
        return move_tiles_result;
    }

    all_equal(ngrid: number[]): boolean {
        for (let index: number = 0; index < this.size; ++index) {
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
        for (let axis: number = 0; axis < this.dim; ++axis) {
            for (let opposite: number = 0; opposite < 2; ++opposite) {
                if (this.simulate_move(axis, Boolean(opposite)).changed) {
                    return false;
                }
            }
        }
        return true;
    }

    check_if_game_over(): boolean {
        return !this.has_available_cells() && this.no_matches_available();
    }

    get_available_cell_indices(): number[] {
        let result: number[] = [];
        this.grid.forEach((val: number, index: number) => {
            if (!val) {
                result.push(index);
            }
        });
        return result;
    }

    generate_random_tile_val(): number {
        return Math.random() < 0.9 ? 1 : 2;
    }

    generate_tile_randomly(): number {
        let available_cell_indices: number[] = this.get_available_cell_indices();
        if (!available_cell_indices.length) {
            return -1;
        }
        let index: number = sample(available_cell_indices);
        let val: number = this.generate_random_tile_val();
        this.grid[index] = val;
        return index;
    }

    snake_fill_grid(): void {
        let axis: number;
        let indices: number[] = new Array(this.dim).fill(0);
        let steps: number[] = new Array(this.dim).fill(1);
        for (let val: number = 0; val < this.size - 1; ++val) {
            this.grid[this.ravel(indices)] = val;
            axis = 0;
            indices[axis] += steps[axis];
            while (indices[axis] === -1 || indices[axis] === this.shape[axis]) {
                if (indices[axis] === -1) {
                    ++indices[axis];
                    steps[axis] = 1;
                } else {
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

    serialize(): GameState {
        return {
            shape: this.shape,
            atom: this.atom,
            view_dim: this.view_dim,
            grid: this.grid,
            total_score: this.total_score,
            game_over: this.game_over,
        }
    }
}


class Recorder {
    game_state_key: string;
    storage: Storage;

    constructor() {
        this.game_state_key = "game_state";
        this.storage = window.localStorage;
    }

    get_game_state(): object | null {
        let state_json: string | null = this.storage.getItem(this.game_state_key);
        return state_json ? JSON.parse(state_json) : null;
    }

    set_game_state(game_state: GameState): void {
        this.storage.setItem(this.game_state_key, JSON.stringify(game_state));
    }

    clear_game_state(): void {
        this.storage.removeItem(this.game_state_key);
    }
}


class Game extends Core {
    c: Constants;
    recorder: Recorder;
    game_container: Element;
    grid_container: Element;
    tile_container: Element;
    score_container: Element;
    game_message: Element;

    constructor(new_game_info: NewGameInfo | null = null) {
        let recorder: Recorder = new Recorder();
        let is_new_game: boolean = true;
        if (new_game_info) {
            super(new_game_info.shape, new_game_info.atom, new_game_info.view_dim);
        } else {
            let recorded: GameState | null = recorder.get_game_state();
            if (recorded) {
                is_new_game = false;
                super(recorded.shape, recorded.atom, recorded.view_dim);
                this.grid = recorded.grid;
                this.previous_grid = this.grid;
                this.total_score = new GiantNumber(recorded.total_score.v);
                this.previous_total_score = this.total_score.g_copy();
                this.game_over = recorded.game_over;
                this.refresh_grid();
            } else {
                super([4, 4], 2, 2);
            }
        }

        this.c = {
            tile_style_limit: 14,
            begin_tiles: 2,
            min_max_text_length: 6,
            page_coverage: 0.8,
            lowest_level_buff: 0.15,  // Relative to cell side length
            buff_gap: 0.3,  // Relative to cell side length
            tile_size_intervals_px: [35, 50, 65, 80, 100],
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
        this.game_container = document.querySelector(".game_container");
        this.grid_container = document.querySelector(".grid_container");
        this.tile_container = document.querySelector(".tile_container");
        this.score_container = document.querySelector(".score_container");
        this.game_message = document.querySelector(".game_message");
        this.init(is_new_game);
    }

    init(is_new_game: boolean): void {
        this.init_consts();
        this.init_client_related_consts();
        this.init_positions();
        this.init_client_related_sizes();
        if (is_new_game) {
            this.add_start_tiles();
            this.update_score();
        } else if (this.total_score) {
            let self: Game = this;
            window.requestAnimationFrame(() => {
                self.update_score_animation(self.total_score);
            });
        }
    }

    get_cell_anchor_point(index: number): number[] {
        let result: number[] = [];
        let indices: number[] = this.unravel(index);
        let view_axis: number;
        let current_unit: number;
        let axis_val: number;
        let axis: number;
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

    get_full_grid_size(): number[] {
        let result: number[] = [];
        let added_width: number = 1.0 + this.c.frame_width;
        this.get_cell_anchor_point(this.size - 1).forEach((axis_val: number) => {
            result.push(axis_val + added_width);
        });
        return result;
    }

    css(id_name: string, rule_command: string): void {
        let style: Element | null = document.querySelector(`#${id_name}`);
        if (style) {
            style.textContent = rule_command;
        } else {
            style = document.createElement("style");
            style.id = id_name;
            style.textContent = rule_command;
            document.head.appendChild(style);
        }
    }

    get_2D_translate_string(translate_list: number[]): string {
        let translate_str_list: string[] = [];
        translate_list.forEach((ratio: number) => {
            translate_str_list.push(`${ratio}em`);
        });
        let translate_str: string = translate_str_list.join(", ");
        return `translate(${translate_str})`;
    }

    get_text_string(val: number): string {
        let power: number = Math.pow(this.atom, val);
        if (power >= Math.pow(10, this.c.max_text_length)) {
            let logarithm: number = Math.log10(this.atom) * val;
            let exponent: number = Math.floor(logarithm);
            if (exponent >= 100) {
                let power_string: string = `${this.atom}^${val}`;
                if (power_string.length > this.c.max_text_length) {
                    return "INF";
                } else {
                    return `${this.atom}^${val}`;
                }
            }
            let decimals: number = this.c.max_text_length - String(exponent).length - 3;
            let coefficient: number = Math.round(Math.pow(10, logarithm - exponent + decimals));
            let coefficient_str: string = String(coefficient);
            return `${coefficient_str[0]}.${coefficient_str.slice(1)}E${exponent}`;
        } else {
            return String(power);
        }
    }

    init_consts(): void {
        this.c.frame_width = Math.ceil(this.dim / this.view_dim - 1) * this.c.buff_gap + this.c.lowest_level_buff;
        let full_grid_size: number[] = this.get_full_grid_size();
        this.c.full_width = full_grid_size[0];
        this.c.full_height = full_grid_size[1];
    }

    init_client_related_consts(): boolean {
        let intervals: number[] = this.c.tile_size_intervals_px;
        let client_width: number = document.body.clientWidth * this.c.page_coverage;
        let tile_size_px: number = client_width / this.c.full_width;
        tile_size_px = Math.max(tile_size_px, intervals[0]);
        tile_size_px = Math.min(tile_size_px, intervals[intervals.length - 1]);
        if (this.c.tile_size_px === tile_size_px) {
            return false;  // No variables changed
        }
        this.c.tile_size_px = tile_size_px;
        this.c.game_field_width_px = tile_size_px * this.c.full_width;
        this.c.game_field_height_px = tile_size_px * this.c.full_height;

        let max_text_length: number = this.c.min_max_text_length;
        intervals.slice(1, intervals.length - 1).forEach((interval: number) => {
            if (tile_size_px > interval) {
                ++max_text_length;
            }
        });
        this.c.max_text_length = max_text_length;
        return true;
    }

    init_positions(): void {
        let anchor_point: number[];
        let transform_str: string;
        let cell: Element;
        let tile_position_commands: string[] = [];
        for (let index: number = 0; index < this.size; ++index) {
            anchor_point = this.get_cell_anchor_point(index);
            transform_str = this.get_2D_translate_string(anchor_point);
            tile_position_commands.push(`.tile_position_${index} {transform: ${transform_str};}`);
            cell = document.createElement("div");
            cell.classList.add("grid_cell", `tile_position_${index}`);
            this.grid_container.appendChild(cell);
        }
        this.css("tile_positions", tile_position_commands.join("\n"));
    }

    init_client_related_sizes(): void {
        this.css("container_size", `.container {
            font-size: ${this.c.tile_size_px}px;
            width: ${this.c.game_field_width_px}px;
        }`);
        this.css("game_container_size", `.game_container {
            width: inherit;
            height: ${this.c.full_height}em;
        }`);
        this.css("grid_and_tile_containers_size", `.grid_container, .tile_container {
            width: ${this.c.full_width}em;
            height: ${this.c.full_height}em;
            border-radius: ${this.c.frame_width}em;
        }`);

        let font_size: number;
        let tile_text_commands: string[] = [];
        for (let text_length: number = 1; text_length <= this.c.max_text_length; ++text_length) {
            font_size = Math.min(0.55, 1.4 / text_length);
            tile_text_commands.push(`.tile_text_length_${text_length} .tile_text {font-size: ${font_size}em;}`);
        }
        this.css("tile_texts", tile_text_commands.join("\n"));
    }

    add_start_tiles(): void {
        let index: number;
        for (let i = 0; i < this.c.begin_tiles; ++i) {
            index = this.generate_tile_randomly();
            this.add_tile(index);
        }
        this.previous_grid = this.grid;
    }

    apply_classes(element: Element, classes: string[]): void {
        element.setAttribute("class", classes.join(" "));
    }

    clear_container(container: Element): void {
        while (container.firstChild) {
            container.removeChild(container.firstChild);
        }
    }

    add_tile(index: number, previous: PreviousType = null, val: number = 0): void {
        if (!val) {
            val = this.grid[index];
        }
        let inner_text: string = this.get_text_string(val);

        let tile: Element = document.createElement("div");
        let classes: string[] = [
            "tile",
            `tile_position_${index}`,
            `tile_val_${Math.min(val, this.c.tile_style_limit)}`,
            `tile_text_length_${inner_text.length}`,
        ];
        this.apply_classes(tile, classes);

        let self: Game = this;
        if (typeof previous === "number") {
            // Make sure that the tile gets rendered in the previous position first
            let previous_classes: string[] = copy<string>(classes);
            previous_classes[1] = `tile_position_${previous}`;
            self.apply_classes(tile, previous_classes);
            window.requestAnimationFrame(() => {
                self.apply_classes(tile, classes);
            });
        } else if (previous) {
            classes.push("tile_merged");
            self.apply_classes(tile, classes);
            // Render the tiles that merged
            previous.forEach((merged_index: number) => {
                self.add_tile(index, merged_index, val - 1);
            });
        } else {
            classes.push("tile_new");
            self.apply_classes(tile, classes);
        }

        let tile_inner: Element = document.createElement("div");
        tile_inner.classList.add("tile_inner");

        let tile_text: Element = document.createElement("div");
        tile_text.classList.add("tile_text");
        tile_text.textContent = inner_text;
        
        tile_inner.appendChild(tile_text);
        tile.appendChild(tile_inner);
        this.tile_container.appendChild(tile);
    }

    update_score(): void {
        this.score_container.textContent = this.total_score.stringify();
    }

    update_score_animation(score: GiantNumber): void {
        this.clear_container(this.score_container);
        this.update_score();
        let addition: Element = document.createElement("div");
        addition.classList.add("score_addition");
        addition.textContent = `+${score.stringify()}`;
        this.score_container.appendChild(addition);
    }

    show_game_over(): void {
        this.game_message.classList.add("game_over_message");
        this.game_message.textContent = "Game over!";
    }

    clear_game_over(): void {
        this.game_message.classList.remove("game_over_message");
    }

    actuate(axis: number, opposite: boolean): boolean {
        let move_tiles_result: MoveTilesResult = this.move_tiles(axis, opposite);
        if (!move_tiles_result.changed) {
            return false;
        }

        let self: Game = this;
        window.requestAnimationFrame(() => {
            self.clear_container(self.tile_container);
            move_tiles_result.link_data.forEach((previous: PreviousType, index: number) => {
                self.add_tile(index, previous);
            });

            let score: GiantNumber = move_tiles_result.score;
            if (score.stringify() !== "0") {
                self.update_score_animation(score);
            }

            if (move_tiles_result.game_over) {
                self.game_over = true;
                self.show_game_over();
                self.delete_saving();
            } else {
                self.save_game();
            }
        });
        return true
    }

    refresh_grid(): void {
        let self: Game = this;
        window.requestAnimationFrame(() => {
            self.clear_container(self.tile_container);
            self.grid.forEach((val: number, index: number) => {
                if (val) {
                    self.add_tile(index);
                }
            });
            self.update_score();
        });
    }

    new_game(): void {
        this.delete_saving();
        for (let index: number = 0; index < this.size; ++index) {
            this.grid[index] = 0;
        }
        this.total_score = new GiantNumber();
        this.previous_total_score = new GiantNumber();
        this.game_over = false;
        this.clear_game_over();
        this.add_start_tiles();
        this.refresh_grid();
    }

    withdraw(): void {
        this.grid = this.previous_grid;
        this.total_score = this.previous_total_score.g_copy();
        this.game_over = false;
        this.clear_game_over();
        this.save_game();
        this.refresh_grid();
    }

    save_game(): void {
        this.recorder.set_game_state(this.serialize());
    }

    delete_saving(): void {
        this.recorder.clear_game_state();
    }

    keyboard_event(event_key: string): void {
        if (this.game_over) {
            return;
        }
        let choice: number = "lkhgdsjiftaw".indexOf(event_key);
        let axis: number = choice % 6;
        let opposite: boolean = choice < 6;
        if (choice < 0 || axis >= this.dim) {
            return;
        }
        this.actuate(axis, opposite);
    }

    window_onresize(): void {
        let changed: boolean = this.init_client_related_consts();
        if (changed) {
            this.init_client_related_sizes();
            this.refresh_grid();
        }
    }

    snake_fill_tiles(): void {
        this.snake_fill_grid();
        this.refresh_grid();
    }
}


var game = new Game();

window.onresize = function() {
    game.window_onresize();
}

document.addEventListener("keypress", (event: Event) => {
    event.preventDefault();
    game.keyboard_event(event.key);
});

document.querySelector(".new_game_button").addEventListener("click", () => {
    game.new_game()
});
document.querySelector(".withdraw_button").addEventListener("click", () => {
    game.withdraw()
});

function new_size_game(shape: number[], atom: number, view_dim: number): void {
    game.delete_saving();
    game.clear_container(game.grid_container);
    game.clear_container(game.tile_container);
    game = new Game({shape: shape, atom: atom, view_dim: view_dim});
}

// TODO: clear all typescript compiling warnings
