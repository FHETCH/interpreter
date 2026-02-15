from pytest import fixture

import numpy as np
from fhetch import parser
from fhetch.data import Scalar, Vector
from fhetch.eval import eval_globals, eval_func
from fhetch.validate import check_name_collision, check_globals_types


@fixture
def prog():
    return parser.Program.parse_file("examples/key_switch.fhetch", parse_all=True).program

def test_parse(prog):
    print(prog)


def test_eval_consts(prog):
    print(eval_globals(prog))


def test_validate(prog):
    check_name_collision(prog)
    global_env = eval_globals(prog)
    check_globals_types(prog, global_env)


def test_eval_func():
    func = parser.Function.parse_string("""
    def Cheb4(x) {
        var T2 = x * x * 2 - 1;
        var T4 = T2 * T2 * 2 - 1;
        return T4;
    }
    """)[0]
    result = eval_func(func, Scalar(1), global_env={})
    assert result == Scalar(1)



# Scalar + Vector tests
def test_scalar_vector_add():
    """Test Scalar + Vector (broadcasting)"""
    s = Scalar(5)
    v = Vector(np.array([1, 2, 3], dtype=np.uint64))
    result = s + v
    assert isinstance(result, Vector)
    assert np.array_equal(result.value, np.array([6, 7, 8]))


def test_scalar_vector_sub():
    """Test Scalar - Vector (broadcasting)"""
    s = Scalar(10)
    v = Vector(np.array([1, 2, 3], dtype=np.uint64))
    result = s - v
    assert isinstance(result, Vector)
    assert np.array_equal(result.value, np.array([9, 8, 7]))


def test_scalar_vector_mul():
    """Test Scalar * Vector (broadcasting)"""
    s = Scalar(3)
    v = Vector(np.array([1, 2, 3], dtype=np.uint64))
    result = s * v
    assert isinstance(result, Vector)
    assert np.array_equal(result.value, np.array([3, 6, 9]))


def test_scalar_vector_with_modulo():
    """Test Scalar operations with Vector and modulo"""
    s = Scalar(10)
    v = Vector(np.array([3, 5, 7], dtype=np.uint64))
    q = Scalar(7)
    
    # 10 + [3,5,7] = [13,15,17] % 7 = [6,1,3]
    result_add = s.add(v, q)
    assert np.array_equal(result_add.value, np.array([6, 1, 3]))
    
    # 10 - [3,5,7] = [7,5,3] % 7 = [0,5,3]
    result_sub = s.sub(v, q)
    assert np.array_equal(result_sub.value, np.array([0, 5, 3]))
    
    # 10 * [3,5,7] = [30,50,70] % 7 = [2,1,0]
    result_mul = s.mul(v, q)
    assert np.array_equal(result_mul.value, np.array([2, 1, 0]))


# Vector + Vector tests (flat)
def test_vector_vector_add():
    """Test Vector + Vector (element-wise)"""
    v1 = Vector(np.array([1, 2, 3], dtype=np.uint64))
    v2 = Vector(np.array([4, 5, 6], dtype=np.uint64))
    result = v1 + v2
    assert np.array_equal(result.value, np.array([5, 7, 9]))


def test_vector_vector_sub():
    """Test Vector - Vector (element-wise)"""
    v1 = Vector(np.array([10, 20, 30], dtype=np.uint64))
    v2 = Vector(np.array([1, 2, 3], dtype=np.uint64))
    result = v1 - v2
    assert np.array_equal(result.value, np.array([9, 18, 27]))


def test_vector_vector_mul():
    """Test Vector * Vector (element-wise)"""
    v1 = Vector(np.array([2, 3, 4], dtype=np.uint64))
    v2 = Vector(np.array([5, 6, 7], dtype=np.uint64))
    result = v1 * v2
    assert np.array_equal(result.value, np.array([10, 18, 28]))


def test_vector_vector_with_modulo():
    """Test Vector * Vector with modulo"""
    v1 = Vector(np.array([10, 20, 30], dtype=np.uint64))
    v2 = Vector(np.array([3, 5, 7], dtype=np.uint64))
    q = Scalar(7)
    
    # [10,20,30] + [3,5,7] = [13,25,37] % 7 = [6,4,2]
    result_add = v1.add(v2, q)
    assert np.array_equal(result_add.value, np.array([6, 4, 2]))
    
    # [10,20,30] * [3,5,7] = [30,100,210] % 7 = [2,2,0]
    result_mul = v1.mul(v2, q)
    assert np.array_equal(result_mul.value, np.array([2, 2, 0]))


# Vector + Scalar tests
def test_vector_scalar_mul():
    """Test Vector * Scalar"""
    v = Vector(np.array([2, 3, 4], dtype=np.uint64))
    s = Scalar(5)
    result = v * s
    assert np.array_equal(result.value, np.array([10, 15, 20]))


def test_vector_int_mul():
    """Test Vector * int"""
    v = Vector(np.array([2, 3, 4], dtype=np.uint64))
    result = v * 3
    assert np.array_equal(result.value, np.array([6, 9, 12]))


def test_vector_scalar_mul_with_modulo():
    """Test Vector * Scalar with modulo"""
    v = Vector(np.array([10, 20, 30], dtype=np.uint64))
    s = Scalar(3)
    q = Scalar(7)
    
    # [10,20,30] * 3 = [30,60,90] % 7 = [2,4,6]
    result = v.mul(s, q)
    assert np.array_equal(result.value, np.array([2, 4, 6]))


# Nested vector tests
def test_nested_vector_flat_vector_mul():
    """Test [[a, b], [c, d]] * [x, y] broadcasts to [[a*x, b*x], [c*y, d*y]]"""
    v1 = Vector(np.array([1, 2, 3], dtype=np.uint64))
    v2 = Vector(np.array([4, 5, 6], dtype=np.uint64))
    nested = Vector(np.array([v1, v2], dtype=object))
    
    flat = Vector(np.array([2, 3], dtype=np.uint64))
    
    result = nested * flat
    
    # First sub-vector: [1,2,3] * 2 = [2,4,6]
    assert np.array_equal(result.value[0].value, np.array([2, 4, 6]))
    # Second sub-vector: [4,5,6] * 3 = [12,15,18]
    assert np.array_equal(result.value[1].value, np.array([12, 15, 18]))


def test_nested_vector_flat_vector_mul_with_modulo():
    """Test [[a, b], [c, d]] * [x, y] with modulo"""
    v1 = Vector(np.array([10, 20, 30], dtype=np.uint64))
    v2 = Vector(np.array([15, 25, 35], dtype=np.uint64))
    nested = Vector(np.array([v1, v2], dtype=object))
    
    flat = Vector(np.array([3, 5], dtype=np.uint64))
    q = Scalar(7)
    
    result = nested.mul(flat, q)
    
    # First sub-vector: [10,20,30] * 3 = [30,60,90] % 7 = [2,4,6]
    assert np.array_equal(result.value[0].value, np.array([2, 4, 6]))
    # Second sub-vector: [15,25,35] * 5 = [75,125,175] % 7 = [5,6,0]
    assert np.array_equal(result.value[1].value, np.array([5, 6, 0]))


def test_nested_vector_nested_vector_mul():
    """Test [[a, b]] * [[c, d]] element-wise nested multiplication"""
    v1 = Vector(np.array([1, 2, 3], dtype=np.uint64))
    v2 = Vector(np.array([4, 5, 6], dtype=np.uint64))
    nested1 = Vector(np.array([v1, v2], dtype=object))
    
    v3 = Vector(np.array([2, 2, 2], dtype=np.uint64))
    v4 = Vector(np.array([3, 3, 3], dtype=np.uint64))
    nested2 = Vector(np.array([v3, v4], dtype=object))
    
    result = nested1 * nested2
    
    # First: [1,2,3] * [2,2,2] = [2,4,6]
    assert np.array_equal(result.value[0].value, np.array([2, 4, 6]))
    # Second: [4,5,6] * [3,3,3] = [12,15,18]
    assert np.array_equal(result.value[1].value, np.array([12, 15, 18]))


def test_nested_vector_nested_vector_mul_with_modulo():
    """Test [[a, b]] * [[c, d]] with modulo"""
    v1 = Vector(np.array([10, 20, 30], dtype=np.uint64))
    v2 = Vector(np.array([15, 25, 35], dtype=np.uint64))
    nested1 = Vector(np.array([v1, v2], dtype=object))
    
    v3 = Vector(np.array([3, 5, 7], dtype=np.uint64))
    v4 = Vector(np.array([2, 4, 6], dtype=np.uint64))
    nested2 = Vector(np.array([v3, v4], dtype=object))
    
    q = Scalar(7)
    result = nested1.mul(nested2, q)
    
    # First: [10,20,30] * [3,5,7] = [30,100,210] % 7 = [2,2,0]
    assert np.array_equal(result.value[0].value, np.array([2, 2, 0]))
    # Second: [15,25,35] * [2,4,6] = [30,100,210] % 7 = [2,2,0]


# Size assertion tests
def test_vector_add_size_mismatch():
    """Test that adding vectors of different sizes raises ValueError"""
    import pytest
    v1 = Vector(np.array([1, 2, 3], dtype=np.uint64))
    v2 = Vector(np.array([4, 5], dtype=np.uint64))
    
    with pytest.raises(ValueError, match="Cannot perform addition: vectors have incompatible shapes"):
        v1 + v2


def test_vector_sub_size_mismatch():
    """Test that subtracting vectors of different sizes raises ValueError"""
    import pytest
    v1 = Vector(np.array([1, 2, 3, 4], dtype=np.uint64))
    v2 = Vector(np.array([4, 5], dtype=np.uint64))
    
    with pytest.raises(ValueError, match="Cannot perform subtraction: vectors have incompatible shapes"):
        v1 - v2


def test_vector_mul_size_mismatch():
    """Test that multiplying vectors of different sizes raises ValueError"""
    import pytest
    v1 = Vector(np.array([1, 2, 3], dtype=np.uint64))
    v2 = Vector(np.array([4, 5, 6, 7], dtype=np.uint64))
    
    with pytest.raises(ValueError, match="Cannot perform multiplication: vectors have incompatible shapes"):
        v1 * v2


def test_vector_add_with_modulo_size_mismatch():
    """Test that adding vectors with modulo and different sizes raises ValueError"""
    import pytest
    v1 = Vector(np.array([10, 20], dtype=np.uint64))
    v2 = Vector(np.array([3, 5, 7], dtype=np.uint64))
    q = Scalar(7)
    
    with pytest.raises(ValueError, match="Cannot perform addition \\(with modulo\\): vectors have incompatible shapes"):
        v1.add(v2, q)


def test_vector_sub_with_modulo_size_mismatch():
    """Test that subtracting vectors with modulo and different sizes raises ValueError"""
    import pytest
    v1 = Vector(np.array([10, 20, 30], dtype=np.uint64))
    v2 = Vector(np.array([3, 5], dtype=np.uint64))
    q = Scalar(7)
    
    with pytest.raises(ValueError, match="Cannot perform subtraction \\(with modulo\\): vectors have incompatible shapes"):
        v1.sub(v2, q)


def test_nested_vector_size_mismatch():
    """Test that nested vectors with size mismatch raise ValueError"""
    import pytest
    v1 = Vector(np.array([1, 2, 3], dtype=np.uint64))
    v2 = Vector(np.array([4, 5, 6], dtype=np.uint64))
    nested1 = Vector(np.array([v1, v2], dtype=object))
    
    # Different outer size
    v3 = Vector(np.array([2, 2, 2], dtype=np.uint64))
    nested2 = Vector(np.array([v3], dtype=object))
    
    with pytest.raises(ValueError, match="Cannot perform multiplication"):
        nested1 * nested2
