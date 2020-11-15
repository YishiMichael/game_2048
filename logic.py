import numpy as np
import time


__all__ = ["push_tiles_using_numpy", "push_tiles_using_loops", "push_tiles_in_vectorized_map"]


link_dtype = np.dtype([("from_index", np.int8), ("to_index", np.int8), ("whether_merge", np.bool_)])


def make_link_data(from_indices, to_indices, whether_merge):
    return np.core.records.fromarrays((from_indices, to_indices, whether_merge), dtype=link_dtype)


def add_opposite_direction_attribute(accumulate_func):
    def wrapper(a, *, axis, opposite):
        aa = a.astype(np.int8)
        if opposite:
            aa = np.flip(aa, axis=axis)
        result = accumulate_func(aa, axis=axis)
        if opposite:
            result = np.flip(result, axis=axis)
        return result
    return wrapper


@add_opposite_direction_attribute
def continual_increment(a, *, axis):
    purely_accumulated = np.add.accumulate(a, axis=axis)
    delta = np.maximum.accumulate(purely_accumulated * np.logical_not(a), axis=axis)
    return purely_accumulated - delta


@add_opposite_direction_attribute
def separate_decrement(a, *, axis):
    # arange -> broadcast
    original_order = np.add.accumulate(np.ones_like(a), axis=axis) - 1
    delta = np.maximum.accumulate(a + original_order * a.astype(np.bool_), axis=axis)
    return delta - original_order


def move_window(a, window_size, *, axis, additional_axis=0):
    shape = list(a.shape)
    shape[axis] -= window_size - 1
    shape.insert(additional_axis, window_size)
    strides = list(a.strides)
    strides.insert(additional_axis, a.strides[axis])
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def stretch(a, stretch_size, *, axis, fill=0, at_tail=True):
    stretched_shape = list(a.shape)
    stretched_shape[axis] = stretch_size
    stretched_part = fill * np.ones(stretched_shape, dtype=a.dtype)
    pair = (a, stretched_part)
    if not at_tail:
        pair = pair[::-1]
    return np.concatenate(pair, axis=axis)


def all_equal(a, *, axis=0):
    return a.max(axis=axis) == a.min(axis=axis)


def ravel_argwhere_along_axis(a, *, axis):
    template_indices = np.moveaxis(np.arange(a.size).reshape(a.shape), axis, -1)
    result = np.where(np.moveaxis(a, axis, -1), template_indices, np.full_like(template_indices, fill_value=-1))
    return result[result != -1]


def move_tiles(vm, *, axis, opposite):
    nonzeros = vm.astype(np.bool_).astype(np.int8)
    nonzeros_order = np.cumsum(nonzeros, axis=axis)
    zeros_order = np.cumsum(1 - nonzeros, axis=axis)
    if opposite:
        nonzeros_order += vm.shape[axis]
    else:
        zeros_order += vm.shape[axis]
    order = np.where(nonzeros, nonzeros_order, zeros_order)
    nvm = np.take_along_axis(vm, np.argsort(order, axis=axis), axis=axis)
    return nvm


def merge_tiles(vm, base, *, axis, opposite):
    basewise_strided = move_window(vm, base, axis=axis, additional_axis=0)
    merged_groups = all_equal(basewise_strided, axis=0)
    merged_groups = stretch(merged_groups, base - 1, axis=axis, fill=False, at_tail=not opposite)
    merged_groups = np.logical_and(merged_groups, vm)
    accumulated_merged_groups = continual_increment(merged_groups, axis=axis, opposite=opposite)
    merged_tiles_mask = accumulated_merged_groups % base == 1
    preserved_boxes_mask = separate_decrement(merged_tiles_mask * base, axis=axis, opposite=opposite) <= 0
    nvm = np.select((merged_tiles_mask, preserved_boxes_mask), (vm + 1, vm))
    repeat_times = np.select((merged_tiles_mask, nvm != 0), (np.full_like(vm, fill_value=base), np.ones_like(vm)))
    repeat_times = repeat_times.flat[ravel_argwhere_along_axis(repeat_times, axis=axis)]
    score = (base ** nvm[merged_tiles_mask != 0]).sum()
    return nvm, repeat_times, score


def push_tiles_using_numpy(vm, base, *, axis, opposite):
    assert base > 1
    vm1 = move_tiles(vm, axis=axis, opposite=opposite)
    vm2, repeat_times, score = merge_tiles(vm1, base, axis=axis, opposite=opposite)
    nvm = move_tiles(vm2, axis=axis, opposite=opposite)

    from_indices = ravel_argwhere_along_axis(vm, axis=axis)
    to_indices = ravel_argwhere_along_axis(nvm, axis=axis).repeat(repeat_times)
    whether_merge = (repeat_times != 1).repeat(repeat_times)
    link_data = make_link_data(from_indices, to_indices, whether_merge)
    link_data.sort(order="from_index")
    return nvm, link_data, score


def push_tiles_along_axis_1(vm, base):
    nvm = np.zeros_like(vm)
    from_indices = []
    to_indices = []
    whether_merge = []
    temp_indices = []
    score = 0
    for i, line in enumerate(vm):
        line_from_indices = []
        line_to_indices = []
        nvm_p = 0
        hold = 0
        count = 0
        temp_indices = []
        for j, val in enumerate(line):
            if val == 0:
                pass
            elif val == hold:
                count += 1
                temp_indices.append(j)
                if count == base:
                    nvm[i, nvm_p] = val + 1
                    line_from_indices.extend(temp_indices)
                    line_to_indices.extend([nvm_p] * count)
                    whether_merge.extend([True] * count)
                    score += base ** (val + 1)
                    nvm_p += 1
                    hold = 0
                    count = 0
                    temp_indices = []
            else:
                nvm[i, nvm_p:nvm_p + count] = hold
                line_from_indices.extend(temp_indices)
                line_to_indices.extend(list(range(nvm_p, nvm_p + count)))
                whether_merge.extend([False] * count)
                nvm_p += count
                hold = val
                count = 1
                temp_indices = [j]
        nvm[i, nvm_p:nvm_p + count] = hold
        line_from_indices.extend(temp_indices)
        line_to_indices.extend(list(range(nvm_p, nvm_p + count)))
        whether_merge.extend([False] * count)
        from_indices.append(line_from_indices)
        to_indices.append(line_to_indices)

    def generator_pair_from_compound_list(compound_list):
        for i, line in enumerate(compound_list):
            for j in line:
                yield i
                yield j

    def compound_list_to_array(compound_list):
        return np.fromiter(generator_pair_from_compound_list(compound_list), dtype=np.int8).reshape((-1, 2))
    
    from_indices = compound_list_to_array(from_indices)
    to_indices = compound_list_to_array(to_indices)
    return nvm, from_indices, to_indices, whether_merge, score


def push_tiles_using_loops(vm, base, *, axis, opposite):
    assert base > 1
    if opposite:
        vm = np.flip(vm, axis=axis)
    original_shape = vm.shape
    vm = np.swapaxes(vm, axis, -1)
    temp_shape = vm.shape
    vm = vm.reshape((-1, original_shape[axis]))
    nvm, from_indices, to_indices, whether_merge, score = push_tiles_along_axis_1(vm, base)
    nvm = nvm.reshape(temp_shape)
    nvm = np.swapaxes(nvm, -1, axis)
    if opposite:
        nvm = np.flip(nvm, axis=axis)

    def recover_axes_and_flatten(indices):
        recovered_indices = np.array(np.unravel_index(np.ravel_multi_index(indices.T, vm.shape), temp_shape))
        recovered_indices[[axis, -1]] = recovered_indices[[-1, axis]]
        if opposite:
            recovered_indices[axis] = original_shape[axis] - 1 - recovered_indices[axis]
        return np.ravel_multi_index(recovered_indices, original_shape)

    from_indices = recover_axes_and_flatten(from_indices)
    to_indices = recover_axes_and_flatten(to_indices)
    link_data = make_link_data(from_indices, to_indices, whether_merge)
    link_data.sort(order="from_index")
    return nvm, link_data, score


def push_tiles_in_vectorized_map(vm, shape, base, *, axis, opposite):
    assert base > 1
    nvm = np.zeros_like(vm)
    from_indices = []
    to_indices = []
    whether_merge = []
    score = 0
    minor_step = np.prod(shape[axis + 1:], dtype=np.int8)
    major_step = minor_step * shape[axis]
    if not opposite:
        for i0 in range(0, vm.size, major_step):
            for i1 in range(i0, i0 + minor_step):
                nvm_p = i1
                hold = 0
                count = 0
                temp_indices = []
                for index in range(i1, i1 + major_step, minor_step):
                    val = vm[index]
                    if val == 0:
                        pass
                    elif val == hold:
                        count += 1
                        temp_indices.append(index)
                        if count == base:
                            nvm[nvm_p] = val + 1
                            from_indices.extend(temp_indices)
                            to_indices.extend([nvm_p] * count)
                            whether_merge.extend([True] * count)
                            score += base ** (val + 1)
                            nvm_p += minor_step
                            hold = 0
                            count = 0
                            temp_indices = []
                    else:
                        nvm[nvm_p:nvm_p + count * minor_step:minor_step] = hold
                        from_indices.extend(temp_indices)
                        to_indices.extend(list(range(nvm_p, nvm_p + count * minor_step, minor_step)))
                        whether_merge.extend([False] * count)
                        nvm_p += count * minor_step
                        hold = val
                        count = 1
                        temp_indices = [index]
                nvm[nvm_p:nvm_p + count * minor_step:minor_step] = hold
                from_indices.extend(temp_indices)
                to_indices.extend(list(range(nvm_p, nvm_p + count * minor_step, minor_step)))
                whether_merge.extend([False] * count)
    else:
        for i0 in range(0, vm.size, major_step):
            for i1 in range(i0, i0 + minor_step):
                nvm_p = i1 + major_step - minor_step
                hold = 0
                count = 0
                temp_indices = []
                for index in range(i1 + major_step - minor_step, i1 - minor_step, -minor_step):
                    val = vm[index]
                    if val == 0:
                        pass
                    elif val == hold:
                        count += 1
                        temp_indices.append(index)
                        if count == base:
                            nvm[nvm_p] = val + 1
                            from_indices.extend(temp_indices)
                            to_indices.extend([nvm_p] * count)
                            whether_merge.extend([True] * count)
                            score += base ** (val + 1)
                            nvm_p -= minor_step
                            hold = 0
                            count = 0
                            temp_indices = []
                    else:
                        nvm[nvm_p:nvm_p - count * minor_step:-minor_step] = hold
                        from_indices.extend(temp_indices)
                        to_indices.extend(list(range(nvm_p, nvm_p - count * minor_step, -minor_step)))
                        whether_merge.extend([False] * count)
                        nvm_p -= count * minor_step
                        hold = val
                        count = 1
                        temp_indices = [index]
                if nvm_p - count * minor_step < 0:
                    nvm[nvm_p::-minor_step] = hold
                else:
                    nvm[nvm_p:nvm_p - count * minor_step:-minor_step] = hold
                from_indices.extend(temp_indices)
                to_indices.extend(list(range(nvm_p, nvm_p - count * minor_step, -minor_step)))
                whether_merge.extend([False] * count)
    link_data = make_link_data(from_indices, to_indices, whether_merge)
    link_data.sort(order="from_index")
    return nvm, link_data, score


def test_equal():
    test_cases = 100
    test_shape = (5, 4, 3, 4)
    test_vm = np.random.randint(4, size=(test_cases, *test_shape))
    test_base = np.random.randint(2, 4, size=test_cases)
    test_axis = np.random.randint(len(test_shape), size=test_cases)
    test_opposite = np.random.randint(2, size=test_cases).astype(np.bool_)
    #test_vm = [np.array([[3, 2, 2, 1], [0, 2, 2, 3], [1, 1, 2, 1], [0, 0, 2, 0]])]
    #test_base = [2]
    #test_axis = [0]
    #test_opposite = [True]
    for vm, base, axis, opposite in zip(test_vm, test_base, test_axis, test_opposite):
        nvm0, link_data0, score0 = push_tiles_using_numpy(vm, base, axis=axis, opposite=opposite)
        nvm1, link_data1, score1 = push_tiles_using_loops(vm, base, axis=axis, opposite=opposite)
        nvm2, link_data2, score2 = push_tiles_in_vectorized_map(vm.reshape(-1), vm.shape, base, axis=axis, opposite=opposite)
        assert np.array_equal(nvm0, nvm1)
        assert np.array_equal(nvm0, nvm2.reshape(vm.shape))
        assert np.array_equal(link_data0, link_data1)
        assert np.array_equal(link_data0, link_data2)
        assert score0 == score1 == score2


def test_time():
    test_cases = 1000
    test_shape = (5, 4, 3, 4)
    test_vm = np.random.randint(4, size=(test_cases, *test_shape))
    test_base = np.random.randint(2, 4, size=test_cases)
    test_axis = np.random.randint(len(test_shape), size=test_cases)
    test_opposite = np.random.randint(2, size=test_cases).astype(np.bool_)
    begin_time = time.time()
    for vm, base, axis, opposite in zip(test_vm, test_base, test_axis, test_opposite):
        _ = push_tiles_using_numpy(vm, base, axis=axis, opposite=opposite)
    end_time = time.time()
    print(end_time - begin_time)
    begin_time = time.time()
    for vm, base, axis, opposite in zip(test_vm, test_base, test_axis, test_opposite):
        _ = push_tiles_using_loops(vm, base, axis=axis, opposite=opposite)
    end_time = time.time()
    print(end_time - begin_time)
    begin_time = time.time()
    for vm, base, axis, opposite in zip(test_vm, test_base, test_axis, test_opposite):
        _ = push_tiles_in_vectorized_map(vm.reshape(-1), vm.shape, base, axis=axis, opposite=opposite)
    end_time = time.time()
    print(end_time - begin_time)


if __name__ == "__main__":
    test_time()
