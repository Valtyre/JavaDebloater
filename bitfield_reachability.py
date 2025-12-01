"""
Bitfield Reachability Analyzer for Dead Code Elimination

Uses the Machine Word Abstract Domain (Bitfield) to detect unreachable branches
based on bitwise operations. For example:
  - Given `if (flags & 0x04)` where flags is 00000⊥00, bit 2 is provably 0,
    so the true-branch is unreachable and can be removed.

This analyzer performs abstract interpretation using the bitfield domain to
track which bits are definitely 0, definitely 1, or unknown (⊥).
"""

import sys
from typing import Iterable, TypeVar
import copy
import jpamb
from jpamb import jvm
from jpamb.jvm.opcode import BitOpr
from dataclasses import dataclass, field
from loguru import logger

# Import the Bitfield abstract domain
from jpamb.abstract_mwa import Bitfield

T = TypeVar("T")


@dataclass
class Stack[T]:
    """Generic stack implementation for operand stack"""
    items: list[T] = field(default_factory=list)

    def __bool__(self) -> bool:
        return len(self.items) > 0

    @classmethod
    def empty(cls):
        return cls([])

    def peek(self) -> T:
        return self.items[-1]

    def pop(self) -> T:
        return self.items.pop(-1)

    def push(self, value: T) -> "Stack[T]":
        self.items.append(value)
        return self

    def copy(self) -> "Stack[T]":
        return Stack(self.items.copy())

    def __str__(self):
        if not self:
            return "ϵ"
        return " ".join(str(v) for v in self.items)


@dataclass(frozen=True)
class PC:
    """Program Counter - identifies a specific bytecode instruction"""
    method: jvm.AbsMethodID
    offset: int

    def __hash__(self):
        return hash((str(self.method), self.offset))

    def __eq__(self, other):
        if not isinstance(other, PC):
            return False
        return str(self.method) == str(other.method) and self.offset == other.offset

    def __add__(self, delta: int) -> "PC":
        return PC(self.method, self.offset + delta)

    def __str__(self):
        return f"{self.method}:{self.offset}"


@dataclass
class Bytecode:
    """Lazy bytecode loader and cache"""
    suite: jpamb.Suite
    methods: dict[jvm.AbsMethodID, list[jvm.Opcode]] = field(default_factory=dict)
    exceptions: dict[jvm.AbsMethodID, list[dict]] = field(default_factory=dict)

    def __getitem__(self, pc: PC) -> jvm.Opcode:
        if pc.method not in self.methods:
            self._load_method(pc.method)
        return self.methods[pc.method][pc.offset]

    def method_length(self, method: jvm.AbsMethodID) -> int:
        if method not in self.methods:
            self._load_method(method)
        return len(self.methods[method])

    def get_exception_handlers(self, method: jvm.AbsMethodID) -> list[int]:
        """Get all exception handler entry points for a method"""
        if method not in self.exceptions:
            self._load_method(method)
        return [exc["handler"] for exc in self.exceptions.get(method, [])]

    def _load_method(self, method: jvm.AbsMethodID):
        """Load method bytecode and exception table"""
        method_json = self.suite.findmethod(method)
        self.methods[method] = [jvm.Opcode.from_json(op) for op in method_json["code"]["bytecode"]]
        self.exceptions[method] = method_json["code"].get("exceptions", [])


# Abstract value type: either a Bitfield or a special marker
@dataclass
class AbstractValue:
    """Wrapper for abstract values in the analysis"""
    bitfield: Bitfield | None = None  # None means non-integer type (reference, etc.)
    is_reference: bool = False

    @staticmethod
    def top() -> "AbstractValue":
        """Unknown integer value"""
        return AbstractValue(bitfield=Bitfield.top())

    @staticmethod
    def of(value: int) -> "AbstractValue":
        """Concrete integer value"""
        return AbstractValue(bitfield=Bitfield.of(value))

    @staticmethod
    def reference() -> "AbstractValue":
        """Reference type (non-integer)"""
        return AbstractValue(is_reference=True)

    def is_definitely_zero(self) -> bool:
        """Check if this value is definitely 0"""
        if self.bitfield is None:
            return False
        val = self.bitfield.to_int()
        return val == 0

    def is_definitely_nonzero(self) -> bool:
        """Check if this value is definitely non-zero (has at least one definite 1 bit)"""
        if self.bitfield is None:
            return False
        return any(b == "1" for b in self.bitfield.bits)

    def join(self, other: "AbstractValue") -> "AbstractValue":
        """Least upper bound of two abstract values"""
        if self.is_reference or other.is_reference:
            return AbstractValue(is_reference=True)
        if self.bitfield is None or other.bitfield is None:
            return AbstractValue.top()
        return AbstractValue(bitfield=self.bitfield.join(other.bitfield))

    def __str__(self):
        if self.is_reference:
            return "ref"
        if self.bitfield is None:
            return "⊤"
        return str(self.bitfield)


@dataclass
class Frame:
    """Execution frame with locals and operand stack"""
    locals: dict[int, AbstractValue]
    stack: Stack[AbstractValue]
    pc: PC

    @staticmethod
    def from_method(method: jvm.AbsMethodID) -> "Frame":
        return Frame({}, Stack.empty(), PC(method, 0))

    def copy(self) -> "Frame":
        return Frame(
            locals=self.locals.copy(),
            stack=self.stack.copy(),
            pc=self.pc
        )

    def __str__(self):
        locals_str = ", ".join(f"{k}:{v}" for k, v in sorted(self.locals.items()))
        return f"<{{{locals_str}}}, {self.stack}, {self.pc}>"


@dataclass
class AbstractState:
    """Abstract state for the analysis"""
    frame: Frame
    reachable: bool = True

    @staticmethod
    def initial(method: jvm.AbsMethodID) -> "AbstractState":
        """Create initial state with all parameters as top (unknown)"""
        frame = Frame.from_method(method)
        for i, param_type in enumerate(method.methodid.params._elements):
            if isinstance(param_type, jvm.Int):
                frame.locals[i] = AbstractValue.top()
            else:
                frame.locals[i] = AbstractValue.reference()
        return AbstractState(frame=frame)

    def copy(self) -> "AbstractState":
        return AbstractState(frame=self.frame.copy(), reachable=self.reachable)

    @property
    def pc(self) -> PC:
        return self.frame.pc

    def __str__(self):
        return f"State({self.frame}, reachable={self.reachable})"


class ReachabilityAnalyzer:
    """
    Abstract interpreter using bitfield domain for reachability analysis.
    
    Tracks which bytecode offsets are reachable and which branches
    can be proven unreachable based on bitwise operations.
    """

    def __init__(self, suite: jpamb.Suite, method: jvm.AbsMethodID):
        self.bc = Bytecode(suite)
        self.method = method
        self.reachable_pcs: set[PC] = set()
        self.unreachable_branches: list[tuple[PC, str]] = []  # (pc, reason)
        self.states: dict[PC, AbstractState] = {}
        self.worklist: set[PC] = set()

    def analyze(self) -> set[int]:
        """
        Run the analysis and return set of reachable bytecode offsets.
        """
        initial = AbstractState.initial(self.method)
        initial_pc = PC(self.method, 0)
        
        self.states[initial_pc] = initial
        self.worklist.add(initial_pc)
        self.reachable_pcs.add(initial_pc)

        # Mark all exception handlers as reachable (conservative)
        # Exception handlers can be reached via exceptions thrown anywhere in their range
        for handler_offset in self.bc.get_exception_handlers(self.method):
            handler_pc = PC(self.method, handler_offset)
            self.reachable_pcs.add(handler_pc)
            if handler_pc not in self.states:
                handler_state = AbstractState.initial(self.method)
                handler_state.frame.pc = handler_pc
                # Exception is on stack at handler entry
                handler_state.frame.stack.push(AbstractValue.reference())
                self.states[handler_pc] = handler_state
                self.worklist.add(handler_pc)

        iterations = 0
        max_iterations = 10000  # Increase for loops

        while self.worklist and iterations < max_iterations:
            iterations += 1
            pc = self.worklist.pop()
            state = self.states[pc]
            
            # Step the state and get successor states
            successors = list(self.step(state))
            
            for succ in successors:
                if not succ.reachable:
                    continue
                    
                succ_pc = succ.pc
                self.reachable_pcs.add(succ_pc)
                
                if succ_pc not in self.states:
                    self.states[succ_pc] = succ
                    self.worklist.add(succ_pc)
                else:
                    # Join with existing state
                    old_state = self.states[succ_pc]
                    new_state = self._join_states(old_state, succ)
                    if self._states_differ(old_state, new_state):
                        self.states[succ_pc] = new_state
                        self.worklist.add(succ_pc)

        return {pc.offset for pc in self.reachable_pcs if pc.method == self.method}

    def _join_states(self, s1: AbstractState, s2: AbstractState) -> AbstractState:
        """Join two abstract states"""
        result = s1.copy()
        # Join locals
        all_keys = set(s1.frame.locals.keys()) | set(s2.frame.locals.keys())
        for k in all_keys:
            v1 = s1.frame.locals.get(k, AbstractValue.top())
            v2 = s2.frame.locals.get(k, AbstractValue.top())
            result.frame.locals[k] = v1.join(v2)
        # Join stacks (assume same length at merge points)
        if len(s1.frame.stack.items) == len(s2.frame.stack.items):
            result.frame.stack = Stack([
                v1.join(v2) for v1, v2 in zip(s1.frame.stack.items, s2.frame.stack.items)
            ])
        return result

    def _states_differ(self, s1: AbstractState, s2: AbstractState) -> bool:
        """Check if two states are different"""
        if s1.frame.locals != s2.frame.locals:
            return True
        if len(s1.frame.stack.items) != len(s2.frame.stack.items):
            return True
        for v1, v2 in zip(s1.frame.stack.items, s2.frame.stack.items):
            if str(v1) != str(v2):
                return True
        return False

    def step(self, state: AbstractState) -> Iterable[AbstractState]:
        """Execute one bytecode instruction and yield successor states"""
        s = state.copy()
        frame = s.frame
        pc = frame.pc
        offset = pc.offset

        try:
            opcode = self.bc[pc]
        except IndexError:
            return

        match opcode:
            # Push constant
            case jvm.Push(value=v):
                if isinstance(v.type, jvm.Int):
                    frame.stack.push(AbstractValue.of(v.value))
                else:
                    frame.stack.push(AbstractValue.reference())
                frame.pc = pc + 1
                yield s

            # Load from local
            case jvm.Load(index=i):
                val = frame.locals.get(i, AbstractValue.top())
                frame.stack.push(val)
                frame.pc = pc + 1
                yield s

            # Store to local
            case jvm.Store(index=i):
                val = frame.stack.pop()
                frame.locals[i] = val
                frame.pc = pc + 1
                yield s

            # Bitwise operations - key for reachability!
            case jvm.Bitwise(operant=op):
                v2 = frame.stack.pop()
                v1 = frame.stack.pop()
                result = self._bitwise_op(v1, v2, op)
                frame.stack.push(result)
                frame.pc = pc + 1
                yield s

            # Binary arithmetic
            case jvm.Binary(operant=op):
                v2 = frame.stack.pop()
                v1 = frame.stack.pop()
                result = self._binary_op(v1, v2, op)
                frame.stack.push(result)
                frame.pc = pc + 1
                yield s

            # Conditional branch on zero comparison
            case jvm.Ifz(condition=cond, target=target):
                v = frame.stack.pop()
                branches = self._evaluate_ifz(v, cond, offset, target)
                
                for branch_target, is_reachable, reason in branches:
                    branch_state = state.copy()
                    branch_state.frame.pc = PC(pc.method, branch_target)
                    branch_state.reachable = is_reachable
                    if not is_reachable:
                        self.unreachable_branches.append((PC(pc.method, branch_target), reason))
                    yield branch_state

            # Two-operand comparison branch
            case jvm.If(condition=cond, target=target):
                v2 = frame.stack.pop()
                v1 = frame.stack.pop()
                branches = self._evaluate_if(v1, v2, cond, offset, target)
                
                for branch_target, is_reachable, reason in branches:
                    branch_state = state.copy()
                    branch_state.frame.pc = PC(pc.method, branch_target)
                    branch_state.reachable = is_reachable
                    if not is_reachable:
                        self.unreachable_branches.append((PC(pc.method, branch_target), reason))
                    yield branch_state

            # Unconditional jump
            case jvm.Goto(target=t):
                frame.pc = PC(pc.method, t)
                yield s

            # Return
            case jvm.Return():
                # Terminal state, no successors
                pass

            # Increment local
            case jvm.Incr(index=i, amount=amt):
                v = frame.locals.get(i, AbstractValue.top())
                if v.bitfield is not None:
                    # After increment, we lose precision
                    frame.locals[i] = AbstractValue.top()
                frame.pc = pc + 1
                yield s

            # Dup
            case jvm.Dup(words=1):
                v = frame.stack.peek()
                frame.stack.push(v)
                frame.pc = pc + 1
                yield s

            # Pop
            case jvm.Pop(words=n):
                for _ in range(n):
                    if frame.stack:
                        frame.stack.pop()
                frame.pc = pc + 1
                yield s

            # Negate
            case jvm.Negate():
                v = frame.stack.pop()
                # Negation loses precision in bitfield domain
                frame.stack.push(AbstractValue.top())
                frame.pc = pc + 1
                yield s

            # Get static/instance field
            case jvm.Get():
                frame.stack.push(AbstractValue.top())
                frame.pc = pc + 1
                yield s

            # TableSwitch - must explore all branches
            case jvm.TableSwitch(default=default, low=low, targets=targets):
                # Pop the switch value
                frame.stack.pop()
                # Yield successors for all targets (we don't know which case matches)
                for target in targets:
                    s_branch = state.copy()
                    s_branch.frame.pc = pc.with_offset(target)
                    yield s_branch
                # Also yield the default target
                s_default = state.copy()
                s_default.frame.pc = pc.with_offset(default)
                yield s_default

            # LookupSwitch - must explore all branches
            case jvm.LookupSwitch(default=default, pairs=pairs):
                # Pop the switch value
                frame.stack.pop()
                # Yield successors for all case targets
                for key, target in pairs:
                    s_branch = state.copy()
                    s_branch.frame.pc = pc.with_offset(target)
                    yield s_branch
                # Also yield the default target
                s_default = state.copy()
                s_default.frame.pc = pc.with_offset(default)
                yield s_default

            # Throw - terminal
            case jvm.Throw():
                pass

            # Default: advance PC, push top for unknown effects
            case _:
                # Unknown opcode - be conservative
                frame.pc = pc + 1
                yield s

    def _bitwise_op(self, v1: AbstractValue, v2: AbstractValue, op: BitOpr) -> AbstractValue:
        """Perform bitwise operation on abstract values"""
        if v1.bitfield is None or v2.bitfield is None:
            return AbstractValue.top()

        match op:
            case BitOpr.And:
                return AbstractValue(bitfield=v1.bitfield & v2.bitfield)
            case BitOpr.Or:
                return AbstractValue(bitfield=v1.bitfield | v2.bitfield)
            case BitOpr.Xor:
                return AbstractValue(bitfield=v1.bitfield ^ v2.bitfield)

        return AbstractValue.top()

    def _binary_op(self, v1: AbstractValue, v2: AbstractValue, op) -> AbstractValue:
        """Perform binary arithmetic - loses precision in bitfield domain"""
        # For addition, subtraction, etc., we lose bit-level precision
        # Could implement more precise transfer functions if needed
        return AbstractValue.top()

    def _evaluate_ifz(self, v: AbstractValue, cond: str, offset: int, target: int
                      ) -> list[tuple[int, bool, str]]:
        """
        Evaluate if-zero condition and return possible branches.
        Returns: [(target_offset, is_reachable, reason), ...]
        
        This is where bitwise guard detection happens!
        """
        fall_through = offset + 1
        branches = []

        # Check if we can prove the condition
        if v.is_definitely_zero():
            # Value is definitely 0
            match cond:
                case "eq":  # if v == 0 → true
                    branches.append((target, True, ""))
                    branches.append((fall_through, False, "value definitely 0, eq always true"))
                case "ne":  # if v != 0 → false
                    branches.append((target, False, "value definitely 0, ne always false"))
                    branches.append((fall_through, True, ""))
                case "lt":  # if v < 0 → false (0 is not < 0)
                    branches.append((target, False, "value definitely 0, lt always false"))
                    branches.append((fall_through, True, ""))
                case "gt":  # if v > 0 → false
                    branches.append((target, False, "value definitely 0, gt always false"))
                    branches.append((fall_through, True, ""))
                case "le":  # if v <= 0 → true
                    branches.append((target, True, ""))
                    branches.append((fall_through, False, "value definitely 0, le always true"))
                case "ge":  # if v >= 0 → true
                    branches.append((target, True, ""))
                    branches.append((fall_through, False, "value definitely 0, ge always true"))
                case _:
                    branches.append((target, True, ""))
                    branches.append((fall_through, True, ""))

        elif v.is_definitely_nonzero():
            # Value has at least one definite 1 bit → definitely non-zero
            match cond:
                case "eq":  # if v == 0 → false
                    branches.append((target, False, "value definitely nonzero, eq always false"))
                    branches.append((fall_through, True, ""))
                case "ne":  # if v != 0 → true
                    branches.append((target, True, ""))
                    branches.append((fall_through, False, "value definitely nonzero, ne always true"))
                case _:
                    # Can't determine sign from just knowing it's nonzero
                    branches.append((target, True, ""))
                    branches.append((fall_through, True, ""))
        else:
            # Can't prove either way - both branches reachable
            branches.append((target, True, ""))
            branches.append((fall_through, True, ""))

        return branches

    def _evaluate_if(self, v1: AbstractValue, v2: AbstractValue, cond: str, 
                     offset: int, target: int) -> list[tuple[int, bool, str]]:
        """Evaluate two-operand comparison condition"""
        fall_through = offset + 1
        
        # For now, be conservative - both branches are reachable
        # Could extend with more precise comparisons
        return [(target, True, ""), (fall_through, True, "")]


def analyze_method(method: jvm.AbsMethodID) -> tuple[set[int], list[tuple[PC, str]]]:
    """
    Analyze a method and return reachable offsets and unreachable branches.
    """
    suite = jpamb.Suite()
    analyzer = ReachabilityAnalyzer(suite, method)
    reachable = analyzer.analyze()
    return reachable, analyzer.unreachable_branches


def main():
    """Entry point for the reachability analyzer"""
    methodid, input_val = jpamb.getcase()
    
    logger.info(f"Analyzing method: {methodid}")
    
    reachable, unreachable = analyze_method(methodid)
    
    # Get total bytecode length
    suite = jpamb.Suite()
    bc = Bytecode(suite)
    total = bc.method_length(methodid)
    all_offsets = set(range(total))
    dead_code = all_offsets - reachable
    
    # Output results
    logger.info(f"Reachable offsets: {sorted(reachable)}")
    logger.info(f"Dead code offsets: {sorted(dead_code)}")
    
    if unreachable:
        logger.warning("Provably unreachable branches:")
        for pc, reason in unreachable:
            logger.warning(f"  {pc}: {reason}")
    
    # Output reachable PCs (for compatibility with existing test harness)
    for offset in sorted(reachable):
        logger.success(f"{offset}")

    # Print wildcard for interpreter compatibility
    print("*")


if __name__ == "__main__":
    main()
