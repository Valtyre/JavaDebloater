# test_polyhedra.py
# Pytest-style smoke tests for the Polyhedra abstraction using the repo's
# canonical abstraction file `jpamb/abstract_mwa_and_poly.py`.

import importlib.util
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
MODULE_PATH = THIS_DIR.parent / "jpamb" / "abstract_mwa_and_poly.py"

spec = importlib.util.spec_from_file_location("abstract_mwa_and_poly", str(MODULE_PATH))
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)  # type: ignore

Polyhedra = mod.Polyhedra


def test_polyhedra_basic_box_and_join_meet():
    # simple box constraints: 0 <= x <= 10
    p1 = Polyhedra()
    p1.add_constraint({"x": 1.0}, 10.0)   # x <= 10
    p1.add_constraint({"x": -1.0}, -0.0)     # x >= 0

    # second polyhedron: x <= 20, y <= 5
    p2 = Polyhedra()
    p2.add_constraint({"x": 1.0}, 20.0)
    p2.add_constraint({"y": 1.0}, 5.0)

    # join -> should create box covering x in [0,20] and y in [0,5]
    j = p1.join(p2)
    rs = repr(j)
    assert "x" in rs
    assert "y" in rs

    # meet -> conjunction of both sets of constraints (still feasible)
    m = p1.meet(p2)
    assert not m.is_bottom()


def test_polyhedra_bottom_detection():
    p3 = Polyhedra()
    p3.add_constraint({}, -1.0)  # 0 <= -1 -> infeasible
    assert p3.is_bottom()
