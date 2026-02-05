from dataclasses import dataclass, field
from enum import StrEnum
from numpy import array
from typing import Optional
from sympy import ntt, intt
from sympy.ntheory.residue_ntheory import nthroot_mod

from ntt import _number_theoretic_transform

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

    def __add__(self, other):
        return ScalarLiteral(self.value + other.value)

    def __sub__(self, other):
        return ScalarLiteral(self.value - other.value)

    def __mul__(self, other):
        return ScalarLiteral(self.value * other.value)

    def __mod__(self, q):
        return ScalarLiteral(self.value % q.value)

    def mmuls(self, other, q):
        """Modular multiplication with Scalar."""
        return (self * other) % q

    def mmulv(self, other, q):
        """Modular multiplication with Vector (element-wise)."""
        return other.mmuls(self, q)


# Scratchpad to store powers of roots of unity
# 
# NOTE: This implementation assumes the root of unity is set the first time an (i)NTT is called 
# for a given size and modulus. The same root is re-used for all subsequent (i)NTTs with the same 
# dimension and modulus.
class _NbTheoryScratchpad:
    
    def __init__(self):
        self.powers_rou = {}
    
    def add_powers_rou(self, modulus: int, ring_dimension: int, rou: int = None):
        if rou is None:
            rou = nthroot_mod(modulus-1, ring_dimension, modulus)
        w = 1
        powers = [w]
        for i in range(1, 2*ring_dimension):
            w = (w * rou) % modulus
            powers.append(w)
        self.powers_rou[(modulus, ring_dimension)] = powers

    def get_powers_rou(self, modulus: int, ring_dimension: int, rou: int = None):
        if not (modulus, ring_dimension) in self.powers_rou: 
            self.add_powers_rou(modulus, ring_dimension, rou=rou)
        return self.powers_rou[(modulus, ring_dimension)]

_nb_theory_scratchpad = _NbTheoryScratchpad()


@dataclass
class VectorLiteral(Expression):
    value: list[int]

    def __repr__(self):
        return repr(self.value)

    def __or__(self, other):
        return VectorLiteral(self.value + other.value)

    def __iter__(self):
        return iter(self.value)

    def __add__(self, other):
        return VectorLiteral([a + b for a, b in zip(self.value, other.value)])

    def __sub__(self, other):
        return VectorLiteral([a - b for a, b in zip(self.value, other.value)])

    def __mod__(self, q):
        return VectorLiteral([a % q for a in self.value])

    def mmuls(self, other, q):
        """Modular multiplication with Scalar."""
        return VectorLiteral([(a * other) % q for a in self.value])

    def mmulv(self, other, q):
        """Modular multiplication with Vector (element-wise)."""
        return VectorLiteral([(a * b) % q for a, b in zip(self.value, other.value)])
    
    def forward_ntt(self, q, rou=None, permutation=None):
        """Forward NTT function"""
        if permutation is None:
            permutation = array(range(len(self.value)))
        q = q.value
        rou2 = (rou*rou)%q if rou is not None else None
        coefficients = [x.value for x in self.value]
        prefactors = _nb_theory_scratchpad.get_powers_rou(q, len(coefficients), rou=rou)
        for (i,coefficient) in enumerate(coefficients[1:]):
            coefficients[i+1] = (prefactors[i+1] * coefficient) % q
        coefficients_ntt = _number_theoretic_transform(coefficients, q, rou=rou2, inverse=False)
        return VectorLiteral(list(map(ScalarLiteral, 
                                      [coefficients_ntt[permutation[i]] for i in range(len(coefficients_ntt))])))
    
    def inverse_ntt(self, q, rou=None, permutation=None):
        """Inverse NTT function"""
        if permutation is None:
            permutation = array(range(len(self.value)))
        q = q.value
        rou2 = (rou*rou)%q if rou is not None else None
        coefficients = [self.value[permutation[i]].value for i in range(len(self.value))]
        coefficients_intt = _number_theoretic_transform(coefficients, q, rou=rou2, inverse=True)
        prefactors = _nb_theory_scratchpad.get_powers_rou(q, len(coefficients))
        for (i,coefficient) in enumerate(coefficients_intt[1:]):
            coefficients_intt[i+1] = (prefactors[2*len(coefficients)-i-1] * coefficient) % q
        return VectorLiteral(list(map(ScalarLiteral, coefficients_intt)))

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
