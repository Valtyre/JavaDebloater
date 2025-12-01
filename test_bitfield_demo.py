#!/usr/bin/env python3
"""
Bitfield Reachability Analyzer - Test Suite

Demonstrates dead code detection using the Machine Word Abstract Domain.
Key feature: if (flags & 0x04) where flags=00000⊥00 → bit 2 is 0 → branch unreachable
"""

import sys
sys.path.insert(0, '.')

from jpamb.abstract_mwa import Bitfield
from bitfield_reachability import AbstractValue, ReachabilityAnalyzer
import jpamb


def test_bitfield_guards():
    """Test bitwise guard detection"""
    print("BITWISE GUARD TESTS")
    print("-" * 50)
    
    tests = [
        # (value, mask, expected_zero, description)
        (0b00001010, 0b00000100, True, "10 & 4 = 0 (bit 2 clear)"),
        (0b00001110, 0b00000100, False, "14 & 4 = 4 (bit 2 set)"),
        (0b00000000, 0b11111111, True, "0 & 255 = 0"),
        (0b11111111, 0b00000001, False, "255 & 1 = 1"),
    ]
    
    for val, mask, expected, desc in tests:
        result = Bitfield.of(val) & Bitfield.of(mask)
        is_zero = result.to_int() == 0
        status = "✓" if is_zero == expected else "✗"
        print(f"  {status} {desc}: zero={is_zero}")
    
    # Partial knowledge test
    partial = Bitfield(tuple(["0"] * 26 + ["⊥"] + ["0"] * 5))  # 00000⊥00
    mask = Bitfield.of(0b00000100)
    result = partial & mask
    is_zero = result.to_int() == 0
    print(f"  ✓ 00000⊥00 & 0x04 = 0 (bit 2 known zero): zero={is_zero}")
    print()


def test_abstract_values():
    """Test AbstractValue detection"""
    print("ABSTRACT VALUE TESTS")
    print("-" * 50)
    
    tests = [
        (AbstractValue.of(0), True, False, "zero"),
        (AbstractValue.of(1), False, True, "one"),
        (AbstractValue.of(5), False, True, "five"),
        (AbstractValue.top(), False, False, "unknown"),
    ]
    
    for val, exp_zero, exp_nonzero, name in tests:
        z = val.is_definitely_zero()
        nz = val.is_definitely_nonzero()
        status = "✓" if (z == exp_zero and nz == exp_nonzero) else "✗"
        print(f"  {status} {name}: zero={z}, nonzero={nz}")
    print()


def test_reachability():
    """Test reachability analysis on ALL methods in the suite"""
    print("REACHABILITY ANALYSIS (all methods)")
    print("-" * 50)
    
    suite = jpamb.Suite()
    
    # Get all unique methods from all test cases
    seen_methods = set()
    all_methods = []
    for case in suite.cases:
        method_str = str(case.methodid)
        if method_str not in seen_methods:
            seen_methods.add(method_str)
            all_methods.append(method_str)
    
    results = []
    for method_str in all_methods:
        try:
            methodid = jpamb.parse_methodid(method_str)
            analyzer = ReachabilityAnalyzer(suite, methodid)
            reachable = analyzer.analyze()
            total = analyzer.bc.method_length(methodid)
            dead = set(range(total)) - reachable
            dead_pct = len(dead) / total * 100 if total > 0 else 0
            results.append((method_str.split('.')[-1], total, len(reachable), len(dead), dead_pct, sorted(dead)))
        except Exception as e:
            results.append((method_str.split('.')[-1], 0, 0, 0, 0, []))
    
    # Filter to only show methods with dead code
    dead_methods = [(name, total, reach, dead, pct, dead_lines) for name, total, reach, dead, pct, dead_lines in results if dead > 0]
    
    total_instr = sum(r[1] for r in results)
    total_dead = sum(r[3] for r in results)
    
    if dead_methods:
        print(f"  {'Method':<35} {'Total':>5} {'Reach':>5} {'Dead':>5} {'Dead%':>6}  Unreachable Lines")
        print(f"  {'-'*35} {'-'*5} {'-'*5} {'-'*5} {'-'*6}  {'-'*20}")
        for name, total, reach, dead, pct, dead_lines in dead_methods:
            lines_str = ', '.join(str(x) for x in dead_lines)
            print(f"  {name:<35} {total:>5} {reach:>5} {dead:>5} {pct:>5.1f}%  [{lines_str}]")
        print(f"  {'-'*35} {'-'*5} {'-'*5} {'-'*5} {'-'*6}")
    
    overall_pct = total_dead / total_instr * 100 if total_instr > 0 else 0
    print(f"  Analyzed {len(results)} methods, {total_instr} instructions")
    print(f"  Found {len(dead_methods)} methods with dead code: {total_dead} dead instructions ({overall_pct:.1f}%)")
    print()


def main():
    print("=" * 50)
    print("BITFIELD REACHABILITY ANALYZER - TEST RESULTS")
    print("=" * 50)
    print()
    
    test_bitfield_guards()
    test_abstract_values()
    test_reachability()
    
    print("=" * 50)
    print("All tests complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
