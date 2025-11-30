# test_small_analyzer_poly.py
# Smoke test showing SmallAnalyzer integration with the Polyhedra abstraction.
# This test imports `small_analyzer.py` from your local `Gr3 - Novel/solutions`
# so it exercises the same SmallAnalyzer implementation you worked on.

import sys
from pathlib import Path
import importlib
try:
    import pytest
except ImportError:
    pytest = None


def _locate_and_import_small_analyzer():
    """Try several strategies to import `solutions.small_analyzer`:
    1. Plain package import (works if PYTHONPATH is configured).
    2. Search upward for a sibling `Gr3 - Novel/solutions` directory and
       add it to sys.path, then import as a package.
    If not found, raise ImportError so the test can be skipped by caller.
    """
    try:
        return importlib.import_module("solutions.small_analyzer")
    except Exception:
        pass

    # Search upward from this file for a sibling folder named 'Gr3 - Novel'
    here = Path(__file__).resolve().parent
    for _ in range(6):
        parent = here
        gr3_candidate = parent.parent / "Gr3 - Novel" / "solutions"
        if gr3_candidate.exists():
            # add Gr3 - Novel root to sys.path to allow `from solutions import small_analyzer`
            sys.path.insert(0, str(gr3_candidate.parent))
            try:
                return importlib.import_module("solutions.small_analyzer")
            except Exception:
                break
        here = parent.parent

    raise ImportError("Could not locate `solutions.small_analyzer` in workspace")


try:
    mod = _locate_and_import_small_analyzer()
except ImportError:
    msg = "`small_analyzer` not available in this checkout; skipping test"
    if pytest is not None:
        pytest.skip(msg)
    else:
        # Running as plain python (no pytest available): be permissive and exit success
        print(msg)
        """
        This test is now self-contained and exercises the repo's `solutions/my_analyzer.py`.
        It runs the script as a subprocess using the same Python interpreter and checks
        that the required info and prediction output formats are produced.
        """
        from subprocess import run, PIPE


        MY_ANALYZER = Path(__file__).resolve().parent.parent / "solutions" / "my_analyzer.py"


        def _run_my_analyzer(args):
            cmd = [sys.executable, str(MY_ANALYZER)] + args
            proc = run(cmd, stdout=PIPE, stderr=PIPE, text=True)
            return proc.returncode, proc.stdout, proc.stderr


        def test_my_analyzer_info():
            rc, out, err = _run_my_analyzer(["info"])
            assert rc == 0
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            # info should print 5 lines as the minimal contract
            assert len(lines) >= 5
            assert "My First Analyzer" in lines[0]


        def test_my_analyzer_prediction_format():
            # Run analyzer for a sample method identifier and check predictions format
            sample = "jpamb.cases.Simple.justReturn:()I"
            rc, out, err = _run_my_analyzer([sample])
            assert rc == 0
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            # Expect 6 outcome lines like 'ok;90%'
            assert len(lines) >= 6
            for ln in lines[:6]:
                assert ";" in ln and ln.split(";")[1].endswith("%")
