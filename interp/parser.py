from pyparsing import pyparsing_common as ppc, Word, Suppress, Keyword, one_of, infix_notation, \
    OpAssoc, DelimitedList, Forward, hexnums, Literal, Optional, Group, cStyleComment, cppStyleComment, \
    ParserElement

from interp import fhetch_ast as ast

ParserElement.enablePackrat()

def KW(name):
    return Suppress(Keyword(name))


def conv_expr(expr):
    match expr:
        case str(var):
            return ast.VarAccess(var)
        case int(value):
            return ast.ScalarLiteral(value)
        case [str(op), operand]:
            return ast.UnaryOperation(op, operand)
        case [lhs, str(op), rhs]:
            return ast.BinaryOperation(ast.BinOp(op), conv_expr(lhs), conv_expr(rhs))
        case [*lhs, str(op), rhs]:
            return ast.BinaryOperation(ast.BinOp(op), conv_expr(lhs), conv_expr(rhs))
        case other:
            return other


Expression = Forward()
# plain_integer is for use in type definitions (lengths, etc.) - stays as int
plain_integer = (Literal("0x") + Word(hexnums)).set_parse_action(lambda s, l, t: int("".join(t), 0)) | ppc.integer
# integer is for use in expressions - becomes ScalarLiteral
integer = plain_integer.copy()
integer.set_parse_action(lambda s, l, t: ast.ScalarLiteral(t[0]))
VecLiteral = Suppress('[') + Optional(DelimitedList(Expression, ',')) + Suppress(']')
VecLiteral.set_parse_action(lambda s, l, t: ast.VectorLiteral(list(map(conv_expr, t))))
FunctionCall = ppc.identifier + Suppress("(") + Group(Optional(DelimitedList(Expression))) + Suppress(")")
FunctionCall.set_parse_action(lambda s, l, t: ast.FunctionCall(t[0], t[1]))
Operand = FunctionCall | ppc.identifier | integer | VecLiteral

signop = one_of("+ -")
multop = one_of("* /")
plusop = one_of("+ -")
shiftop = one_of("<< >>")
catop = one_of("||")

Expression << infix_notation(
    Operand,
    [
        (signop, 1, OpAssoc.RIGHT),
        (multop, 2, OpAssoc.LEFT),
        (plusop, 2, OpAssoc.LEFT),
        (shiftop, 2, OpAssoc.LEFT),
        (catop, 2, OpAssoc.LEFT),
    ],
)

Expression.set_parse_action(lambda s, l, t: conv_expr(t[0]))

Type = Forward()
ScalarType = Keyword("u32") | Keyword("u64") | Keyword("i32") | Keyword("i64")
ScalarType.set_parse_action(lambda s, l, t: ast.ScalarType(*t))
VectorType = Suppress("Vector") + Suppress("<") + Type + Suppress(",") + plain_integer + Suppress(">")
VectorType.set_parse_action(lambda s, l, t: ast.VecType(*t))
MRPType = Suppress("MRP") + Suppress("<") + ScalarType + Suppress(",") + plain_integer + Suppress(",") + Expression + Suppress(">")
MRPType.set_parse_action(lambda s, l, t: ast.MRPType(*t))
Type << (ScalarType | VectorType | MRPType)

Prime = Keyword('prime') + ppc.identifier("name") + Suppress("=") + Expression("value") + Suppress(";")
Prime.set_parse_action(lambda s, l, t: ast.Constant(*t))
Primes = Keyword('primes') + ppc.identifier("name") + Suppress("=") + Expression("values") + Suppress(";")
Primes.set_parse_action(lambda s, l, t: ast.Constant(*t))
ConstValue = KW("const") + ppc.identifier("name") + Suppress(":") + Type("type") + Suppress("=") + Expression("value") + Suppress(";")
ConstValue.set_parse_action(lambda s, l, t: ast.Constant(t[1], t[0], t[2]))

Constant = Prime | Primes | ConstValue

Argument = ppc.identifier + Optional(Suppress(":") + Type)
Argument.set_parse_action(lambda s, l, t: ast.Argument(*t))
FunctionArguments = Suppress("(") + Optional(DelimitedList(Argument)) + Suppress(")")
VarDef = (
    KW("var")
    + ppc.identifier("var_name")
    + (Optional(Suppress(":") + Type))("type")
    + Suppress("=")
    + Expression("value")
    + Optional(Suppress("(") + KW("mod") + (ppc.identifier | integer)("modulus") + Suppress(")"))
)
VarDef.set_parse_action(lambda s, l, t: ast.VarDefinition.from_tokens(*t))
Return = (KW("return") + Expression("ret_val")).set_parse_action(lambda s, l, t: ast.Return(t["ret_val"]))
# using Group and t[0][0] is a hack to reuse the FunctionCall in CallStatement
CallStatement = Group(FunctionCall)
CallStatement.set_parse_action(lambda s, l, t: ast.CallStatement(t[0][0]))
Statement = (VarDef | Return | CallStatement) + Suppress(";")
Function = KW("def") + ppc.identifier("name") + FunctionArguments("args") + Suppress("{") + (Statement * (None, None))("statements") + Suppress("}")
Function.set_parse_action(lambda s, l, t: ast.Function(t.name, t.args.as_list(), t.statements))

Program = ((Constant | Function)* (None, None))("program")
Program.ignore(cStyleComment).ignore(cppStyleComment)
Program.set_parse_action(lambda s, l, t: ast.Program(t.as_list()))
