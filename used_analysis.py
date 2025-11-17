from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Set, List

import jpamb
from jpamb import jvm


class UseStatus(Enum):
    NOT_USED = auto()        
    PROBABLY_USED = auto()   
    USED = auto()            


def join_status(a: UseStatus, b: UseStatus) -> UseStatus:
    order = [UseStatus.NOT_USED, UseStatus.PROBABLY_USED, UseStatus.USED]
    return order[max(order.index(a), order.index(b))]


def reachable_pcs_for_method(
    suite: jpamb.Suite,
    method: jvm.AbsMethodID,
) -> Set[int]:
    opcodes = list(suite.method_opcodes(method))
    n = len(opcodes)
    if n == 0:
        return set()

    worklist: List[int] = [0]
    visited: Set[int] = set()

    while worklist:
        pc = worklist.pop()
        if pc in visited or pc < 0 or pc >= n:
            continue
        visited.add(pc)

        op = opcodes[pc]

        match op:
            case jvm.Goto(target=t):
                succs = [t]

            case jvm.Ifz(target=t, condition=_):
                succs = [t, pc + 1]

            case jvm.If(target=t, condition=_):
                succs = [t, pc + 1]

            case jvm.Return(type=_):
                succs = []         

            case _:
                succs = [pc + 1]   

        for s in succs:
            if 0 <= s < n and s not in visited:
                worklist.append(s)

    return visited


@dataclass
class MethodUsage:
    status: UseStatus
    reachable_pcs: Set[int]


def compute_usage(suite: jpamb.Suite) -> Dict[jvm.AbsMethodID, MethodUsage]:
    all_methods: Set[jvm.AbsMethodID] = set()
    for method, _input in suite.case_methods():
        all_methods.add(method)

    entry_methods: Set[jvm.AbsMethodID] = set()
    for method, _input in suite.case_methods():
        entry_methods.add(method)

    reachable: Dict[jvm.AbsMethodID, Set[int]] = {}
    calls: Dict[jvm.AbsMethodID, Set[jvm.AbsMethodID]] = {m: set() for m in all_methods}

    for m in all_methods:
        pcs = reachable_pcs_for_method(suite, m)
        reachable[m] = pcs

        opcodes = list(suite.method_opcodes(m))
        for pc in pcs:
            op = opcodes[pc]
            match op:
                case jvm.InvokeStatic(method=callee):
                    calls[m].add(callee)
                case _:
                    pass

    usage: Dict[jvm.AbsMethodID, MethodUsage] = {}
    for m in all_methods:
        pcs = reachable.get(m, set())
        if not pcs:
            usage[m] = MethodUsage(status=UseStatus.NOT_USED, reachable_pcs=pcs)
        elif m in entry_methods:
            usage[m] = MethodUsage(status=UseStatus.USED, reachable_pcs=pcs)
        else:
            usage[m] = MethodUsage(status=UseStatus.PROBABLY_USED, reachable_pcs=pcs)

    return usage


def main() -> None:
    import sys

    print("RUNTIME PYTHON:", sys.version)
    print("EXECUTABLE:", sys.executable)

    suite = jpamb.Suite()
    usage_info = compute_usage(suite)

    for m, info in usage_info.items():
        match info.status:
            case UseStatus.USED:
                tag = "is used"
            case UseStatus.PROBABLY_USED:
                tag = "is probably used"
            case UseStatus.NOT_USED:
                tag = "is not used"

        print(f"{m} -> {tag} (at PCs: {sorted(info.reachable_pcs)})")


if __name__ == "__main__":
    main()
