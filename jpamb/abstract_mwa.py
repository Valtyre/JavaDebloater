# Machine word abstract domain - Bitfield domain
# Based on https://en.wikipedia.org/wiki/Abstract_interpretation#Machine_word_abstract_domains
# 
# The bitfield domain treats each bit in a machine word separately using 3-valued logic: {0, 1, ⊥}
# where ⊥ represents "unknown" (could be either 0 or 1).
# 
# Concretization function γ:
#   γ(0) = {0}
#   γ(1) = {1}  
#   γ(⊥) = {0, 1}
#
# Abstraction function α:
#   α({0}) = 0
#   α({1}) = 1
#   α({0,1}) = ⊥
#   α({}) = ⊥

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

# Three-valued logic constants
BitValue = Literal["0", "1", "⊥"]


@dataclass(frozen=True)
class Bitfield:
    """
    Bitfield domain for machine word abstraction.
    
    A machine word of width n is treated as an array of n abstract bit values,
    each from the set {0, 1, ⊥} where:
    - 0: definitely zero
    - 1: definitely one
    - ⊥: unknown (could be 0 or 1)
    
    Supports bitwise operations using Kleene's three-valued logic.
    """
    bits: tuple[BitValue, ...]
    bit_width: int = 32
    
    def __init__(self, bits: tuple[BitValue, ...] | str = "⊥", bit_width: int = 32):
        """
        Create a bitfield.
        
        Args:
            bits: Either a tuple of bit values or a single value to replicate
            bit_width: Width of the machine word in bits
        """
        if isinstance(bits, str):
            object.__setattr__(self, 'bits', tuple([bits] * bit_width))
        else:
            object.__setattr__(self, 'bits', bits)
        object.__setattr__(self, 'bit_width', len(self.bits) if isinstance(bits, tuple) else bit_width)
    
    @staticmethod
    def of(value: int, bit_width: int = 32) -> "Bitfield":
        """
        Abstract a concrete integer value to a bitfield.
        
        This implements the abstraction function α for singleton sets:
        α({value}) = bitfield representation of value
        """
        bits_list = []
        for i in range(bit_width):
            bit_val = (value >> i) & 1
            bits_list.append("1" if bit_val else "0")
        return Bitfield(tuple(reversed(bits_list)), bit_width)
    
    @staticmethod
    def top(bit_width: int = 32) -> "Bitfield":
        """Create a top element (all bits unknown)"""
        return Bitfield("⊥", bit_width)
    
    def to_int(self) -> int | None:
        """
        Concretize bitfield to integer if all bits are known.
        Returns None if any bit is unknown.
        """
        if "⊥" in self.bits:
            return None
        result = 0
        for bit in self.bits:
            result = (result << 1) | (1 if bit == "1" else 0)
        return result
    
    def join(self, other: "Bitfield") -> "Bitfield":
        """
        Least upper bound (⊔): merge two bitfields.
        Different bits become unknown (⊥).
        
        This computes the smallest bitfield that covers both inputs.
        """
        if self.bit_width != other.bit_width:
            raise ValueError("Bitfields must have same width")
        
        result = []
        for a, b in zip(self.bits, other.bits):
            if a == b:
                result.append(a)
            else:
                result.append("⊥")
        return Bitfield(tuple(result), self.bit_width)
    
    def meet(self, other: "Bitfield") -> "Bitfield":
        """
        Greatest lower bound (⊓): intersection of two bitfields.
        Returns most precise information from both.
        """
        if self.bit_width != other.bit_width:
            raise ValueError("Bitfields must have same width")
        
        result = []
        for a, b in zip(self.bits, other.bits):
            if a == "⊥":
                result.append(b)
            elif b == "⊥":
                result.append(a)
            elif a == b:
                result.append(a)
            else:
                # Contradiction: one bit is 0, other is 1
                raise ValueError("Incompatible bitfields in meet")
        return Bitfield(tuple(result), self.bit_width)
    
    def __and__(self, other: "Bitfield") -> "Bitfield":
        """
        Bitwise AND using three-valued logic.
        
        Truth table (from Wikipedia):
            A ∧ B | 0  ⊥  1
            ------+---------
            0     | 0  0  0
            ⊥     | 0  ⊥  ⊥
            1     | 0  ⊥  1
        """
        if self.bit_width != other.bit_width:
            raise ValueError("Bitfields must have same width")
        
        result = []
        for a, b in zip(self.bits, other.bits):
            # AND truth table
            if a == "0" or b == "0":
                result.append("0")
            elif a == "1" and b == "1":
                result.append("1")
            else:
                result.append("⊥")
        return Bitfield(tuple(result), self.bit_width)
    
    def __or__(self, other: "Bitfield") -> "Bitfield":
        """
        Bitwise OR using three-valued logic.
        
        Truth table (from Wikipedia):
            A ∨ B | 0  ⊥  1
            ------+---------
            0     | 0  ⊥  1
            ⊥     | ⊥  ⊥  1
            1     | 1  1  1
        """
        if self.bit_width != other.bit_width:
            raise ValueError("Bitfields must have same width")
        
        result = []
        for a, b in zip(self.bits, other.bits):
            # OR truth table
            if a == "1" or b == "1":
                result.append("1")
            elif a == "0" and b == "0":
                result.append("0")
            else:
                result.append("⊥")
        return Bitfield(tuple(result), self.bit_width)
    
    def __xor__(self, other: "Bitfield") -> "Bitfield":
        """
        Bitwise XOR using three-valued logic.
        
        XOR is 1 if bits differ, 0 if same, ⊥ if unknown.
        """
        if self.bit_width != other.bit_width:
            raise ValueError("Bitfields must have same width")
        
        result = []
        for a, b in zip(self.bits, other.bits):
            if a == "⊥" or b == "⊥":
                result.append("⊥")
            elif a != b:
                result.append("1")
            else:
                result.append("0")
        return Bitfield(tuple(result), self.bit_width)
    
    def __invert__(self) -> "Bitfield":
        """
        Bitwise NOT using three-valued logic.
        
        Truth table (from Wikipedia):
            NOT(A) | Result
            -------+-------
            0      | 1
            ⊥      | ⊥
            1      | 0
        """
        result = tuple("1" if b == "0" else "0" if b == "1" else "⊥" for b in self.bits)
        return Bitfield(result, self.bit_width)
    
    def lshift(self, n: int) -> "Bitfield":
        """Left shift by n positions, filling with zeros"""
        if n >= self.bit_width:
            return Bitfield(tuple(["0"] * self.bit_width), self.bit_width)
        new_bits = self.bits[n:] + tuple(["0"] * n)
        return Bitfield(new_bits, self.bit_width)
    
    def rshift(self, n: int) -> "Bitfield":
        """Right shift by n positions, filling with zeros (logical shift)"""
        if n >= self.bit_width:
            return Bitfield(tuple(["0"] * self.bit_width), self.bit_width)
        new_bits = tuple(["0"] * n) + self.bits[:-n]
        return Bitfield(new_bits, self.bit_width)
    
    def is_top(self) -> bool:
        """Check if this is the top element (all bits unknown)"""
        return all(b == "⊥" for b in self.bits)
    
    def is_bottom(self) -> bool:
        """
        Bitfield domain has no bottom element.
        Every bitfield pattern is realizable.
        """
        return False
    
    def __repr__(self) -> str:
        return "".join(self.bits)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Bitfield):
            return False
        return self.bits == other.bits
    
    def __hash__(self) -> int:
        return hash(self.bits)
