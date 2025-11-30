# Abstract domains for machine word analysis
# Based on interval analysis and bitfield domains from the course material

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Set, Literal, Dict, List, Tuple
import math

# constants for three-valued logic
THREE_VALUED_0 = "0"
THREE_VALUED_1 = "1" 
THREE_VALUED_BOT = "⊥"  # unknown bit value

@dataclass(frozen=True)
class Interval:
    """Basic interval domain [low, high]"""
    low: float = -math.inf
    high: float = math.inf

    @staticmethod
    def of(*values: int):
        # create interval from concrete values
        if not values:
            return Interval()
        return Interval(min(values), max(values))

    def is_bottom(self) -> bool:
        # bottom means impossible/empty (low > high)
        return self.low > self.high
    
    def is_top(self) -> bool:
        return self.low == -math.inf and self.high == math.inf

    def __repr__(self) -> str:
        if self.is_bottom():
            return "⊥"
        if self.is_top():
            return "(-∞, +∞)"
        return f"[{int(self.low) if self.low != -math.inf else '-∞'}, {int(self.high) if self.high != math.inf else '+∞'}]"

    def join(self, other: "Interval") -> "Interval":
        # least upper bound - take widest range
        if self.is_bottom(): 
            return other
        if other.is_bottom(): 
            return self
        return Interval(min(self.low, other.low), max(self.high, other.high))

    def meet(self, other: "Interval") -> "Interval":
        # intersection of intervals
        low = max(self.low, other.low)
        high = min(self.high, other.high)
        return Interval(low, high)

    def widen(self, other: "Interval") -> "Interval":
        # widening operator for loop termination
        low = self.low if self.low <= other.low else -math.inf
        high = self.high if self.high >= other.high else math.inf
        return Interval(low, high)

    def __add__(self, other: "Interval") -> "Interval":
        if self.is_bottom() or other.is_bottom():
            return Interval(1, 0)  # bottom
        return Interval(self.low + other.low, self.high + other.high)

    def __sub__(self, other: "Interval") -> "Interval":
        if self.is_bottom() or other.is_bottom():
            return Interval(1, 0)
        return Interval(self.low - other.high, self.high - other.low)

    def __mul__(self, other: "Interval") -> "Interval":
        if self.is_bottom() or other.is_bottom():
            return Interval(1, 0)
        a, b, c, d = self.low, self.high, other.low, other.high
        # need to check all combinations
        return Interval(min(a*c, a*d, b*c, b*d), max(a*c, a*d, b*c, b*d))

    def __neg__(self) -> "Interval":
        if self.is_bottom():
            return Interval(1, 0)
        return Interval(-self.high, -self.low)
    
    def __and__(self, other: "Interval") -> "Interval":
        # bitwise AND - conservative approximation
        if self.is_bottom() or other.is_bottom():
            return Interval(1, 0)
        
        if self.low < 0 or other.low < 0:
            # negative numbers make this tricky
            return Interval(-max(abs(self.high), abs(other.high)), 
                          max(self.high, other.high))
        
        return Interval(0, min(self.high, other.high))
    
    def __or__(self, other: "Interval") -> "Interval":
        if self.is_bottom() or other.is_bottom():
            return Interval(1, 0)
        
        if self.low < 0 or other.low < 0:
            return Interval(-max(abs(self.high), abs(other.high)), 
                          max(self.high, other.high))
        
        return Interval(0, max(self.high, other.high))
    
    def __xor__(self, other: "Interval") -> "Interval":
        if self.is_bottom() or other.is_bottom():
            return Interval(1, 0)
        
        if self.low < 0 or other.low < 0:
            return Interval(-max(abs(self.high), abs(other.high)), 
                          max(self.high, other.high))
        
        return Interval(0, max(self.high, other.high))
    
    def lshift(self, bits: int) -> "Interval":
        if self.is_bottom():
            return Interval(1, 0)
        return Interval(self.low * (2 ** bits), self.high * (2 ** bits))
    
    def rshift(self, bits: int) -> "Interval":
        if self.is_bottom():
            return Interval(1, 0)
        return Interval(self.low // (2 ** bits), self.high // (2 ** bits))


@dataclass(frozen=True, order=True)
class SignedInterval(Interval):
    """Signed interval with bit width (default 32-bit)"""
    bit_width: int = 32
    
    def __post_init__(self):
        min_val = -(2 ** (self.bit_width - 1))
        max_val = 2 ** (self.bit_width - 1) - 1
        
        # clamp to valid range
        if not (self.low == -math.inf or self.low == math.inf):
            object.__setattr__(self, 'low', max(self.low, min_val))
        if not (self.high == -math.inf or self.high == math.inf):
            object.__setattr__(self, 'high', min(self.high, max_val))


@dataclass(frozen=True, order=True)
class UnsignedInterval(Interval):
    """Unsigned interval [0, 2^n - 1]"""
    bit_width: int = 32
    
    def __post_init__(self):
        min_val = 0
        max_val = 2 ** self.bit_width - 1
        
        if not (self.low == -math.inf or self.low == math.inf):
            object.__setattr__(self, 'low', max(self.low, min_val))
        if not (self.high == -math.inf or self.high == math.inf):
            object.__setattr__(self, 'high', min(self.high, max_val))


@dataclass(frozen=True)
class Bitfield:
    """Bitfield with 3-valued logic: 0, 1, or ⊥ (unknown)"""
    bits: tuple = ()
    bit_width: int = 32
    
    def __init__(self, bits: tuple | str = "⊥", bit_width: int = 32):
        if isinstance(bits, str):
            object.__setattr__(self, 'bits', tuple([bits] * bit_width))
        else:
            object.__setattr__(self, 'bits', bits)
        object.__setattr__(self, 'bit_width', bit_width)
    
    @staticmethod
    def of(value: int, bit_width: int = 32) -> "Bitfield":
        # convert concrete value to bitfield
        bits_list = []
        for i in range(bit_width):
            bit_val = (value >> i) & 1
            bits_list.append("1" if bit_val else "0")
        return Bitfield(tuple(reversed(bits_list)), bit_width)
    
    def _bit_op_3valued(self, op: str, other: "Bitfield") -> "Bitfield":
        # 3-valued logic operations
        result = []
        for a, b in zip(self.bits, other.bits):
            if op == "AND":
                table = {
                    ("0", "0"): "0", ("0", "1"): "0", ("0", "⊥"): "0",
                    ("1", "0"): "0", ("1", "1"): "1", ("1", "⊥"): "⊥",
                    ("⊥", "0"): "0", ("⊥", "1"): "⊥", ("⊥", "⊥"): "⊥",
                }
            elif op == "OR":
                table = {
                    ("0", "0"): "0", ("0", "1"): "1", ("0", "⊥"): "⊥",
                    ("1", "0"): "1", ("1", "1"): "1", ("1", "⊥"): "1",
                    ("⊥", "0"): "⊥", ("⊥", "1"): "1", ("⊥", "⊥"): "⊥",
                }
            elif op == "XOR":
                table = {
                    ("0", "0"): "0", ("0", "1"): "1", ("0", "⊥"): "⊥",
                    ("1", "0"): "1", ("1", "1"): "0", ("1", "⊥"): "⊥",
                    ("⊥", "0"): "⊥", ("⊥", "1"): "⊥", ("⊥", "⊥"): "⊥",
                }
            result.append(table.get((a, b), "⊥"))
        return Bitfield(tuple(result), self.bit_width)
    
    def __and__(self, other: "Bitfield") -> "Bitfield":
        return self._bit_op_3valued("AND", other)
    
    def __or__(self, other: "Bitfield") -> "Bitfield":
        return self._bit_op_3valued("OR", other)
    
    def __xor__(self, other: "Bitfield") -> "Bitfield":
        return self._bit_op_3valued("XOR", other)
    
    def __invert__(self) -> "Bitfield":
        # flip bits (0->1, 1->0, ⊥ stays ⊥)
        result = tuple("1" if b == "0" else "0" if b == "1" else "⊥" for b in self.bits)
        return Bitfield(result, self.bit_width)
    
    def join(self, other: "Bitfield") -> "Bitfield":
        # merge bitfields - different bits become unknown
        result = []
        for a, b in zip(self.bits, other.bits):
            if a == b:
                result.append(a)
            else:
                result.append("⊥")
        return Bitfield(tuple(result), self.bit_width)
    
    def is_bottom(self) -> bool:
        return False
    
    def __repr__(self) -> str:
        return "".join(self.bits)


@dataclass
class Polyhedra:
    """A lightweight (conservative) polyhedra abstraction.

    Representation:
    - A list of linear constraints of the form coeffs <= rhs, where
      `coeffs` is a dict mapping variable names to coefficients.
    - `_is_bottom` flag for infeasible polyhedron.

    Notes:
    - This is a simple, conservative implementation intended as a novel
      abstraction extension for machine-word analysis exercises.
    - `join` is approximated via variable-wise interval projection
      (i.e., an over-approximating box) to keep implementation simple.
    """

    constraints: List[Tuple[Dict[str, float], float]] = None
    vars: Set[str] = None
    _is_bottom: bool = False

    def __init__(self, constraints: List[Tuple[Dict[str, float], float]] | None = None):
        self.constraints = list(constraints) if constraints else []
        self.vars = set()
        for coeffs, _ in self.constraints:
            self.vars.update(coeffs.keys())
        self._is_bottom = False

    @staticmethod
    def top() -> "Polyhedra":
        return Polyhedra()

    @staticmethod
    def bottom() -> "Polyhedra":
        p = Polyhedra()
        p._is_bottom = True
        return p

    def is_bottom(self) -> bool:
        return self._is_bottom

    def add_constraint(self, coeffs: Dict[str, float], rhs: float) -> None:
        """Add a linear constraint sum(coeffs[var]*var) <= rhs.

        If an inconsistency is detected (e.g., 0 <= negative), mark bottom.
        """
        # trivial contradiction: no vars but rhs negative
        if not coeffs and rhs < 0:
            self._is_bottom = True
            return

        # store constraint and update var set
        self.constraints.append((dict(coeffs), float(rhs)))
        self.vars.update(coeffs.keys())

        # quick single-variable contradiction check
        if len(coeffs) == 1:
            (v, a), = coeffs.items()
            # if coefficient is zero the constraint reduces to 0 <= rhs
            # which is infeasible when rhs < 0
            if a == 0 and rhs < 0:
                self._is_bottom = True

    def meet(self, other: "Polyhedra") -> "Polyhedra":
        """Intersection: combine constraints from both polyhedra."""
        if self.is_bottom() or other.is_bottom():
            return Polyhedra.bottom()
        # Intersection is simply the conjunction of all constraints.
        # We return a new Polyhedra whose constraints are the union of both
        # constraint sets. We perform a cheap consistency check using
        # single-variable projections: if any variable has an empty interval
        # the polyhedron is infeasible (marked bottom).
        combined = Polyhedra(self.constraints + other.constraints)
        bounds = combined._compute_bounds()
        for v, (lo, hi) in bounds.items():
            # If both lower and upper bounds are present and inconsistent
            # we can declare the intersection infeasible.
            if lo is not None and hi is not None and lo > hi:
                return Polyhedra.bottom()
        return combined

    def join(self, other: "Polyhedra") -> "Polyhedra":
        """Over-approximate join by projecting both polyhedra to per-variable
        intervals and returning the box (interval constraints) that covers both.
        This is a conservative but cheap approximation of convex hull.
        """
        if self.is_bottom():
            return other
        if other.is_bottom():
            return self

        # The true convex hull (least upper bound) of two polyhedra is
        # another convex polyhedron; computing it precisely requires linear
        # programming / convex-hull algorithms. For this assignment we use a
        # cheap but sound over-approximation: project both polyhedra onto
        # each variable and compute the box (interval) that covers both
        # projections. The result is a set of single-variable constraints
        # that over-approximate the convex hull.
        bounds_a = self._compute_bounds()
        bounds_b = other._compute_bounds()

        all_vars = set(bounds_a.keys()) | set(bounds_b.keys())
        box = Polyhedra()
        for v in all_vars:
            la, ha = bounds_a.get(v, (None, None))
            lb, hb = bounds_b.get(v, (None, None))

            # compute conservative lower and upper bounds for v
            if la is not None and lb is not None:
                lo = min(la, lb)
            else:
                lo = la if la is not None else lb

            if ha is not None and hb is not None:
                hi = max(ha, hb)
            else:
                hi = ha if ha is not None else hb

            # encode the interval bounds as linear constraints
            # v >= lo  <=>  -v <= -lo  (represented as coeff {v:-1} <= -lo)
            # v <= hi  <=>   v <= hi   (coeff {v:1} <= hi)
            if lo is not None:
                box.add_constraint({v: -1.0}, -float(lo))
            if hi is not None:
                box.add_constraint({v: 1.0}, float(hi))

        return box

    def _compute_bounds(self) -> Dict[str, Tuple[float | None, float | None]]:
        """Compute simple per-variable bounds from single-var constraints.

        Returns mapping var -> (low, high) where None indicates unknown.
        Only constraints of the form a*v <= b (single variable) are used.
        """
        bounds: Dict[str, Tuple[float | None, float | None]] = {}
        # Only use single-variable constraints to compute simple lower/upper
        # bounds. Multi-var constraints are ignored in this projection (they
        # could tighten bounds but projecting them precisely requires more
        # machinery).
        for coeffs, rhs in self.constraints:
            if not coeffs:
                # constraint is 0 <= rhs; infeasible if rhs < 0
                if rhs < 0:
                    return {v: (1.0, 0.0) for v in self.vars}  # mark infeasible
                continue

            if len(coeffs) == 1:
                (v, a), = coeffs.items()
                lo, hi = bounds.get(v, (None, None))
                # a * v <= rhs
                # if a > 0  => v <= rhs / a
                # if a < 0  => v >= rhs / a
                if a > 0:
                    ub = rhs / a
                    hi = ub if hi is None else min(hi, ub)
                elif a < 0:
                    lb = rhs / a
                    lo = lb if lo is None else max(lo, lb)
                bounds[v] = (lo, hi)

        return bounds

    def assume_equal(self, var: str, value: float) -> None:
        """Assume variable `var` equals `value` by adding two constraints."""
        self.add_constraint({var: 1.0}, float(value))
        self.add_constraint({var: -1.0}, -float(value))

    def __repr__(self) -> str:
        if self.is_bottom():
            return "Polyhedra(⊥)"
        if not self.constraints:
            return "Polyhedra(top)"
        parts = []
        for coeffs, rhs in self.constraints:
            terms = []
            for v, a in coeffs.items():
                terms.append(f"{a}*{v}")
            parts.append(" + ".join(terms) + f" <= {rhs}")
        return "Polyhedra(" + "; ".join(parts) + ")"


# Example usage (not executed during import):
def _example_polyhedra_usage():
    p1 = Polyhedra()
    p1.add_constraint({"x": 1.0}, 10.0)   # x <= 10
    p1.add_constraint({"x": -1.0}, -0.0)  # x >= 0

    p2 = Polyhedra()
    p2.add_constraint({"x": 1.0}, 20.0)   # x <= 20
    p2.add_constraint({"y": 1.0}, 5.0)    # y <= 5

    j = p1.join(p2)
    return p1, p2, j
