from pytest import fixture

from interp import parser
from interp.eval import eval_consts, eval_func
from interp.fhetch_ast import ScalarLiteral
from interp.validate import check_name_collision, check_globals_types


@fixture
def prog():
    return parser.Program.parse_file("examples/key_switch.fhetch", parse_all=True).program

def test_parse(prog):
    print(prog)


def test_eval_consts(prog):
    eval_consts(prog)
    print(prog)


def test_validate(prog):
    check_name_collision(prog)
    eval_consts(prog)
    check_globals_types(prog)


def test_eval_func():
    func = parser.Function.parse_string("""
    def Cheb4(x) {
        var T2 = x * x * 2 - 1;
        var T4 = T2 * T2 * 2 - 1;
        return T4;
    }
    """)[0]
    result = eval_func(func, ScalarLiteral(1), global_env={})
    assert result == ScalarLiteral(1)
