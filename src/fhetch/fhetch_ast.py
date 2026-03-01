from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional


def _eval_const_scalar(parse_result):
    match parse_result:
        case int(operand) | [int(operand)] | ["+", int(operand)]:
            return operand
        case ["-", operand]:
            return -operand
        case [lhs, "+", rhs]:
            lhs = _eval_const_scalar(lhs)
            rhs = _eval_const_scalar(rhs)
            return lhs + rhs
        case [lhs, "-", rhs]:
            lhs = _eval_const_scalar(lhs)
            rhs = _eval_const_scalar(rhs)
            return lhs - rhs
        case [lhs, "*", rhs]:
            lhs = _eval_const_scalar(lhs)
            rhs = _eval_const_scalar(rhs)
            return lhs * rhs
        case [lhs, "<<", rhs]:
            lhs = _eval_const_scalar(lhs)
            rhs = _eval_const_scalar(rhs)
            return lhs << rhs
        case [lhs, ">>", rhs]:
            lhs = _eval_const_scalar(lhs)
            rhs = _eval_const_scalar(rhs)
            return lhs >> rhs
        case other:
            raise TypeError(other)


class Expression: ...


class Type: ...


@dataclass
class Constant:
    type: str | Type
    name: str
    value: Expression

    def __repr__(self):
        return f"{self.type} {self.name} = {self.value};"


class ScalarType(Type, StrEnum):
    U32 = "u32"
    U64 = "u64"
    I32 = "i32"
    I64 = "i64"


@dataclass
class VecType(Type):
    inner: Type
    length: int

    def __repr__(self):
        return f"Vector<{self.inner}, {self.length}>"


@dataclass
class MRPType:
    scalar: ScalarType
    length: int
    modulus: Expression

    def __repr__(self):
        return f"MRP<{self.scalar}, {self.length}, {self.modulus}>"


@dataclass
class Argument:
    name: str
    type: Optional[ScalarType | VecType | MRPType] = None

    def __repr__(self):
        if self.type is None:
            return self.name
        return f"{self.name}: {self.type}"


class Instruction: ...


@dataclass
class VarAccess(Expression):
    var_name: str

    def __repr__(self):
        return self.var_name


class BinOp(StrEnum):
    Add = "+"
    Sub = "-"
    Mul = "*"
    Concat = "||"
    Shl = "<<"
    Shr = ">>"


@dataclass
class BinaryOperation(Expression):
    operation: BinOp
    lhs: Expression
    rhs: Expression

    def __repr__(self):
        return f"{self.lhs} {self.operation} {self.rhs}"

@dataclass
class UnaryOperation(Expression):
    operation: str
    operand: Expression


@dataclass
class FunctionCall(Expression):
    function_name: str
    arguments: list[Expression]

    def __repr__(self):
        args = ", ".join(map(repr, self.arguments))
        return f"{self.function_name}({args})"


class Literal(Expression): ...


@dataclass
class ScalarLiteral(Expression):
    value: int

    def __repr__(self):
        return repr(self.value)


@dataclass
class StringLiteral(Expression):
    value: str

    def __repr__(self):
        return f'"{self.value}"'


@dataclass
class VectorLiteral(Expression):
    value: list[Expression]

    def __repr__(self):
        return repr(self.value)

    def __or__(self, other):
        return VectorLiteral(self.value + other.value)

    def __iter__(self):
        return iter(self.value)


@dataclass
class VarDefinition(Instruction):
    var_name: str
    type: Optional[ScalarType | VecType | MRPType]
    definition: Expression
    modulus: Optional[VarAccess | int] = None

    @classmethod
    def from_tokens(cls, *tokens):
        match tokens:
            case (str(name), expr):
                return cls(name, None, expr)
            case (str(name), expr, str(q)):
                return cls(name, None, expr, VarAccess(q))
            case (str(name), expr, int(q)):
                return cls(name, None, expr, q)
            case (str(name), ty, expr):
                return cls(name, ty, expr)
            case (str(name), ty, expr, str(q)):
                return cls(name, ty, expr, VarAccess(q))
            case (str(name), ty, expr, int(q)):
                return cls(name, ty, expr, q)
            case other:
                raise TypeError("Cannot construct VarDefinition from", other)

    def __repr__(self):
        typ = "" if self.type is None else f": {self.type}"
        return f"var {self.var_name}{typ} = {self.definition}"


@dataclass
class Return(Instruction):
    return_value: Expression

    def __repr__(self):
        return f"return {self.return_value}"


@dataclass
class CallStatement(Instruction):
    call: FunctionCall


@dataclass
class Function:
    name: str
    args: list[Argument]
    body: list[Instruction]

    def __repr__(self):
        args = ", ".join(map(repr, self.args))
        res = f"def {self.name}({args}) {{\n"
        for instr in self.body:
            res += f"    {instr};\n"
        return res + "  }"


@dataclass
class Program:
    items: list[Constant | Function] = field(default_factory=list)

    def get(self, name):
        for item in self.items:
            if item.name == name:
                return item
        return None

    def __repr__(self):
        items = "\n".join(f"  {item}" for item in self.items)
        return f"Program {{\n{items}\n}}"
