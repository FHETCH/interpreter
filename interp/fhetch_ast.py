from __future__ import annotations
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
            return lhs + rhs
        case [lhs, "*", rhs]:
            lhs = _eval_const_scalar(lhs)
            rhs = _eval_const_scalar(rhs)
            return lhs + rhs
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

def builtin_add(lhs, rhs, q):
    assert isinstance(lhs, VectorLiteral) and isinstance(rhs, VectorLiteral)
    assert len(lhs.value) == len(rhs.value)
    return (lhs + rhs) % q


def builtin_sub(lhs, rhs, q):
    assert isinstance(lhs, VectorLiteral) and isinstance(rhs, VectorLiteral)
    assert len(lhs.value) == len(rhs.value)
    return (lhs - rhs) % q


def builtin_mul(lhs, rhs, q):
    if isinstance(rhs, ScalarLiteral):
        return lhs.mmuls(rhs, q)
    elif isinstance(rhs, VectorLiteral):
        return lhs.mmulv(rhs, q)
    # integers??
    return (lhs * rhs) % q


def verify_vectors_size(lhs: VectorLiteral, rhs: VectorLiteral):
    if isinstance(rhs, VectorLiteral):
        if len(lhs.value) != len(rhs.value):
            raise Exception("vector size mismatch")



def verify_literal_types(
    lhs: VectorLiteral | ScalarLiteral, rhs: VectorLiteral | ScalarLiteral
):
    """
    Verify that the types of two literals are compatible.
    
    Note: This function assumes that vectors passed here are 1-dimensional (flat).
    """
    if isinstance(lhs, VectorLiteral):
        lhs_type = lhs.type.inner
    else:
        lhs_type = lhs.type

    if isinstance(rhs, VectorLiteral):
        rhs_type = rhs.type.inner
    else:
        rhs_type = rhs.type

    if (
        lhs_type != None
        and rhs_type != None
        and lhs_type != rhs_type
    ):
        raise TypeError(f"Type mismatch: cannot perform operation between {lhs_type} and {rhs_type}")
    pass


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
    inner: ScalarType|VecType
    length: int

    def __repr__(self):
        return f"Vector<{self.inner}, {self.length}>"
    
    def __eq__(self, other):
        if not isinstance(other, VecType):
            return False
        return self.inner == other.inner and self.length == other.length


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
    type: ScalarType = None

    def __repr__(self):
        return repr(self.value)

    def __neg__(self):
        return ScalarLiteral(-self.value)

    def __add__(self, other):
        verify_literal_types(self, other)
        return ScalarLiteral(self.value + other.value)

    def __sub__(self, other):
        verify_literal_types(self, other)
        return ScalarLiteral(self.value - other.value)

    def __mul__(self, other):
        verify_literal_types(self, other)
        return ScalarLiteral(self.value * other.value)

    def __mod__(self, q):
        return ScalarLiteral(self.value % q.value)

    def mmuls(self, other, q):
        """Modular multiplication with Scalar."""
        verify_literal_types(self, other)
        return (self * other) % q

    def mmulv(self, other, q):
        """Modular multiplication with Vector (element-wise)."""
        verify_literal_types(self, other)
        return other.mmuls(self, q)


@dataclass
class VectorLiteral(Expression):
    value: list[ScalarLiteral|VectorLiteral|int]
    type: VecType = None

    def __repr__(self):
        return repr(self.value)

    def __or__(self, other):
        return VectorLiteral(self.value + other.value)

    def __iter__(self):
        return iter(self.value)

    def __neg__(self):
        return VectorLiteral([-a for a in self.value])

    def __add__(self, other):
        verify_vectors_size(self, other)
        verify_literal_types(self, other)
        return VectorLiteral([a + b for a, b in zip(self.value, other.value)])

    def __sub__(self, other):
        verify_vectors_size(self, other)
        verify_literal_types(self, other)
        return VectorLiteral([a - b for a, b in zip(self.value, other.value)])

    def __mod__(self, q):
        return VectorLiteral([a % q for a in self.value])

    def mmuls(self, other:ScalarLiteral, q):
        """Modular multiplication with Scalar."""
        #verify_vectors_size(self, other)
        verify_literal_types(self, other)
        
        # TODO: self.type.inner is ScalarType
        if isinstance(self.value[0], ScalarLiteral):   
            return VectorLiteral([(a * other) % q for a in self.value])
        
        #TODO: recursive type building
        return VectorLiteral([builtin_mul(a, other, q) for a in self.value])

    def mmulv(self, other:VectorLiteral, q):
        """Modular multiplication with Vector (element-wise)."""
        #verify_vectors_size(self, other)
        if isinstance(self.value[0], ScalarLiteral) and isinstance(other[0], ScalarLiteral):
            verify_literal_types(self, other)
            return VectorLiteral([(a * other) % q for a in self.value])
        return VectorLiteral([builtin_mul(a, b, q) % q for a, b in zip(self.value, other.value)])


@dataclass
class VarDefinition(Instruction):
    var_name: str
    type: Optional[ScalarType | VecType | MRPType]
    definition: Expression
    modulus: Optional[VarAccess | ScalarLiteral] = None

    @classmethod
    def from_tokens(cls, *tokens):
        match tokens:
            case (str(name), expr):
                return cls(name, None, expr)
            case (str(name), expr, str(q)):
                return cls(name, None, expr, VarAccess(q))
            case (str(name), expr, ScalarLiteral(q)):
                return cls(name, None, expr, ScalarLiteral(q))
            case (str(name), ty, expr):
                return cls(name, ty, expr)
            case (str(name), ty, expr, str(q)):
                return cls(name, ty, expr, VarAccess(q))
            case (str(name), ty, expr, ScalarLiteral(q)):
                return cls(name, ty, expr, ScalarLiteral(q))
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
