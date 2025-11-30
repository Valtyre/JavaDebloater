#Test for Machine Word Abstract Domain - Bitfield

import sys
import os
import importlib.util
from jpamb.abstract_mwa import Bitfield

def test_bitfield_operations():
    """Test the Bitfield domain with 3-valued logic"""
    print("\n=== Testing Bitfield Domain ===\n")
    
    # Basic bitfield creation
    b1 = Bitfield.of(5, 8)
    b2 = Bitfield.of(3, 8)
    print(f"b1 = {b1} (5 in binary)")
    print(f"b2 = {b2} (3 in binary)")
    
    # Bitwise operations
    print(f"\nAND: {b1} & {b2} = {b1 & b2}")
    print(f"OR:  {b1} | {b2} = {b1 | b2}")
    print(f"XOR: {b1} ^ {b2} = {b1 ^ b2}")
    print(f"NOT: ~{b1} = {~b1}")
    
    # Unknown bits
    b_unknown = Bitfield("⊥", 8)
    print(f"\nUnknown bitfield: {b_unknown}")
    
    # Operations with unknown bits
    print(f"\nOperations with unknown bits:")
    b_partial = Bitfield(tuple(["0", "0", "0", "0", "⊥", "1", "⊥", "1"]), 8)  # 0000⊥1⊥1
    b_concrete = Bitfield.of(15, 8)  # 00001111
    print(f"Partial: {b_partial}")
    print(f"Concrete: {b_concrete}")
    print(f"AND: {b_partial} & {b_concrete} = {b_partial & b_concrete}")
    print(f"OR:  {b_partial} | {b_concrete} = {b_partial | b_concrete}")
    print(f"NOT: ~{b_partial} = {~b_partial}")
    
    # Join operation
    b0 = Bitfield.of(0, 8)
    b5 = Bitfield.of(5, 8)
    print(f"\nJoin {b0} and {b5} = {b0.join(b5)}")


def test_lattice_operations():
    """Test lattice operations (join and meet)"""
    print("\n=== Testing Lattice Operations ===\n")
    
    # Join (least upper bound)
    b1 = Bitfield.of(5, 8)  # 00000101
    b2 = Bitfield.of(3, 8)  # 00000011
    joined = b1.join(b2)
    print(f"Join of {b1} and {b2} = {joined}")
    
    # Meet (greatest lower bound) with unknown bits
    b_partial1 = Bitfield(tuple(["0", "0", "0", "0", "⊥", "1", "0", "1"]), 8)
    b_partial2 = Bitfield(tuple(["0", "0", "0", "0", "0", "1", "⊥", "1"]), 8)
    meet = b_partial1.meet(b_partial2)
    print(f"\nMeet of {b_partial1} and {b_partial2} = {meet}")
    
    # Top element
    top = Bitfield.top(8)
    print(f"\nTop element: {top}")
    print(f"Is top? {top.is_top()}")


def test_shift_operations():
    """Test shift operations"""
    print("\n=== Testing Shift Operations ===\n")
    
    b = Bitfield.of(5, 8)  # 00000101
    print(f"Original: {b}")
    print(f"Left shift by 2: {b.lshift(2)}")
    print(f"Right shift by 1: {b.rshift(1)}")
    
    # Shift with unknown bits
    b_unknown = Bitfield(tuple(["0", "0", "0", "0", "⊥", "1", "⊥", "1"]), 8)
    print(f"\nWith unknown bits: {b_unknown}")
    print(f"Left shift by 1: {b_unknown.lshift(1)}")


def test_concretization():
    """Test concretization (gamma function)"""
    print("\n=== Testing Concretization ===\n")
    
    # Fully concrete bitfield
    b_concrete = Bitfield.of(42, 8)
    print(f"Bitfield: {b_concrete}")
    print(f"Concretized to int: {b_concrete.to_int()}")
    
    # Partially unknown bitfield
    b_abstract = Bitfield(tuple(["0", "0", "⊥", "⊥", "1", "0", "1", "0"]), 8)
    print(f"\nBitfield with unknowns: {b_abstract}")
    print(f"Concretized to int: {b_abstract.to_int()} (None because of ⊥)")


def test_three_valued_logic():
    """Test 3-valued logic truth tables"""
    print("\n=== 3-Valued Logic Truth Tables ===\n")
    
    print("AND Truth Table:")
    print("  A │ B │ A∧B")
    print("────┼───┼────")
    vals = [("0", "0"), ("0", "1"), ("0", "⊥"), 
            ("1", "0"), ("1", "1"), ("1", "⊥"),
            ("⊥", "0"), ("⊥", "1"), ("⊥", "⊥")]
    
    for a, b in vals:
        bf_a = Bitfield(tuple([a]), 1)
        bf_b = Bitfield(tuple([b]), 1)
        result = (bf_a & bf_b).bits[0]
        print(f"  {a} │ {b} │ {result}")
    
    print("\nOR Truth Table:")
    print("  A │ B │ A∨B")
    print("────┼───┼────")
    for a, b in vals:
        bf_a = Bitfield(tuple([a]), 1)
        bf_b = Bitfield(tuple([b]), 1)
        result = (bf_a | bf_b).bits[0]
        print(f"  {a} │ {b} │ {result}")
    
    print("\nNOT Truth Table:")
    print("  A │ ¬A")
    print("────┼────")
    for a in ["0", "⊥", "1"]:
        bf_a = Bitfield(tuple([a]), 1)
        result = (~bf_a).bits[0]
        print(f"  {a} │ {result}")


if __name__ == "__main__":
    print("=" * 70)
    print("  Machine Word Abstract Domain - Bitfield Test Suite")
    print("=" * 70)
    
    test_bitfield_operations()
    test_lattice_operations()
    test_shift_operations()
    test_concretization()
    test_three_valued_logic()
    
    print("\n" + "=" * 70)
    print("  All tests completed")
    print("=" * 70 + "\n")
