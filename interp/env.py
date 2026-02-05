import sys

from interp.fhetch_ast import Constant, ScalarLiteral, VectorLiteral, builtin_add, builtin_mul, builtin_sub

def builtin_write(obj):
    match obj:
        case ScalarLiteral(x):
            sys.stdout.buffer.write(x.to_bytes(4, "little"))
        case VectorLiteral(v):
            for x in v:
                builtin_write(x)
        case Constant(_, _, value):
            builtin_write(value)
        case other:
            print("Unknown type", type(other), file=sys.stderr)
            print(other)





def default_global():
    return {
        "write": builtin_write,
        "sr_addp": builtin_add,
        "sr_subp": builtin_sub,
        "sr_mulp": builtin_mul,
    }
