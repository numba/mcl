import numpy as np
import mcl
import pytest
import time
from mcl.machine_types import i32, i64, intp, memref
from mcl.vm import Type
from mcl.ndarray import Array, DType, Int32

def to_custom_indices(val):
    if not isinstance(val, np.ndarray):
        return val

    res_val = Array(dtype=DType(Int32), data=memref.alloc(tuple(map(intp, val.shape)), i32))
    for idx in np.ndindex(val.shape):
        res_val[idx] = i32(int(val[idx]))
    return res_val

@pytest.fixture(scope='module')
def setup_arrays():
    np_array = np.random.rand(1000, 1000)
    return np_array, to_custom_indices(np_array)

def benchmark_function(func):
    start_time = time.time()
    func()
    end_time = time.time()
    return end_time - start_time

lambda_getitem_func = lambda arr, indices: arr[*indices]

@pytest.mark.parametrize("operation", [
    ("getitem_single_element", (500, 500)),
    ("getitem_row", (500, slice(None))),
    ("getitem_column", (slice(None), 500)),
    ("getitem_simple_slice", (slice(100, 200), slice(100, 200))),
    ("getitem_fancy_indexing_single_array", (np.random.randint(100, size=(5)),)),
    # TODO: Fix this error, seems to be an issue with indexing, needs it's test too
    # ("getitem_fancy_indexing_2d_array", (np.random.randint(100, size=(10, 20)), np.random.randint(200, size=(10, 20)))),
    # TODO: Fix this error, seems to be an issue with indexing, needs it's test too
    # ("getitem_fancy_indexing_2d_array", (np.random.randint(100, size=(10, 20)), np.random.randint(200, size=(10, 10, 20)))),
])
def test_getitem_indexing_operations(setup_arrays, operation):
    name, indices = operation
    np_array, custom_array = setup_arrays
    
    np_time = benchmark_function(lambda: lambda_getitem_func(np_array, indices))
    print(f"NumPy - {name}: {np_time:.6f} seconds")
    
    custom_indices = tuple([to_custom_indices(ary) for ary in indices])
    
    custom_time = benchmark_function(lambda: lambda_getitem_func(custom_array, custom_indices))
    print(f"CustomArray - {name}: {custom_time:.6f} seconds")
    
    if np_time > 0:
        factor = custom_time / np_time
        print(f"Factor for {name} (CustomArray / NumPy): {factor:.2f}")
    else:
        print(f"NumPy result for {name} is zero, cannot compute performance factor.")
    
    assert np_time > 0
    assert custom_time > 0

def lambda_setitem_func(arr, indices, val):
    arr[*indices] = val

@pytest.mark.parametrize("operation", [
    ("setitem_single_element", (500, 500), 10),
    ("setitem_row", (500, slice(None)), 10),
    ("setitem_column", (slice(None), 500), 10),
    ("setitem_simple_slice", (slice(100, 200), slice(100, 200)), 10),

    # TODO: Array setitem is not supported in mcl right now
    # Once it's supported, need benchmarks with array values
    # ("setitem_fancy_indexing_single_array", (np.random.randint(100, size=(5)),), 10),
    # ("setitem_fancy_indexing_2d_array", (np.random.randint(100, size=(10, 20)), np.random.randint(200, size=(10, 20)))),
    # ("setitem_fancy_indexing_2d_array", (np.random.randint(100, size=(10, 20)), np.random.randint(200, size=(10, 10, 20)))),
])
def test_setitem_indexing_operations(setup_arrays, operation):
    name, indices, val = operation
    np_array, custom_array = setup_arrays
    
    np_time = benchmark_function(lambda: lambda_setitem_func(np_array, indices, val))
    print(f"NumPy - {name}: {np_time:.6f} seconds")
    
    custom_indices = tuple([to_custom_indices(ary) for ary in indices])
    custom_val = i32(val)
    
    custom_time = benchmark_function(lambda: lambda_setitem_func(custom_array, custom_indices, custom_val))
    print(f"CustomArray - {name}: {custom_time:.6f} seconds")
    
    if np_time > 0:
        factor = custom_time / np_time
        print(f"Factor for {name} (CustomArray / NumPy): {factor:.2f}")
    else:
        print(f"NumPy result for {name} is zero, cannot compute performance factor.")
    
    assert np_time > 0
    assert custom_time > 0
