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


