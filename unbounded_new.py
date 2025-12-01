from typing import Iterable, TypeVar
import copy
import jpamb
from jpamb import jvm
from dataclasses import dataclass
from loguru import logger
import sign
from sign import SignSet



T = TypeVar("T")

@dataclass
class Stack[T]:
    items: list[T]

    def __bool__(self) -> bool:
        return len(self.items) > 0

    @classmethod
    def empty(cls):
        return cls([])

    def peek(self) -> T:
        return self.items[-1]

    def pop(self) -> T:
        return self.items.pop(-1)

    def push(self, value):
        self.items.append(value)
        return self

    def __str__(self):
        if not self:
            return "ϵ"
        return "".join(f"{v}" for v in self.items)

@dataclass(frozen=True)
class PC:
    method: jvm.AbsMethodID
    offset: int

    def __hash__(self):
        return hash((str(self.method), self.offset))

    def __eq__(self, other):
        if not isinstance(other, PC):
            return False
        return str(self.method) == str(other.method) and self.offset == other.offset

    def __add__(self, delta):
        return PC(self.method, self.offset + delta)

    def __str__(self):
        return f"{self.method}:{self.offset}"
    
    
@dataclass
class Bytecode:
    suite: jpamb.Suite
    methods: dict[jvm.AbsMethodID, list[jvm.Opcode]]

    def __getitem__(self, pc: PC) -> jvm.Opcode:
        try:
            opcodes = self.methods[pc.method]
        except KeyError:
            opcodes = list(self.suite.method_opcodes(pc.method))
            self.methods[pc.method] = opcodes

        return opcodes[pc.offset]


@dataclass
class Frame:
    locals: dict[int, SignSet | jvm.Boolean]
    stack: Stack
    pc: PC

    def from_method(method: jvm.AbsMethodID) -> "Frame":
        return Frame({}, Stack.empty(), PC(method, 0))
    
    def __str__(self):
        locals = ", ".join(f"{k}:{v}" for k, v in self.locals.items())
        return f"<{{{locals}}}, {self.stack}, {self.pc}>"


@dataclass
class A: # Abstract State
    frames: Stack[Frame]
    pc: PC

    def initialstate_from_method(methodid: jvm.AbsMethodID) -> "A":
        frame: Frame = Frame.from_method(methodid)
        for i, v in enumerate(methodid.methodid.params._elements):
            if isinstance(v, jvm.Int):
                frame.locals[i] = SignSet.top()
            else: 
                frame.locals[i] = v
        return A( Stack.empty().push(frame), PC(methodid, 0))
    
    def __ior__(self, other: "A") -> None:
        assert self.pc == other.pc, "Cannot merge states with different program counters"
        for i in range(len(self.frames.peek().locals)-1):
            if isinstance(other.frames.peek().locals[i], SignSet) and isinstance(self.frames.peek().locals[i], SignSet):
                self.frames.peek().locals[i] |= other.frames.peek().locals[i]
            else:
                self.frames.peek().locals[i] = other.frames.peek().locals[i]
        for i in range(len(self.frames.peek().stack.items)-1):
            if isinstance(other.frames.peek().stack.items[i], SignSet) and isinstance(self.frames.peek().stack.items[i], SignSet):
                self.frames.peek().stack.items[i] |= other.frames.peek().stack.items[i]
            else:
                self.frames.peek().stack.items[i] = other.frames.peek().stack.items[i]
        return self
    
    def advance_pc(self, delta: int = 1):
        self.pc = self.pc + delta
        self.frames.peek().pc = self.pc

    def set_pc(self, new_pc: PC):
        self.pc = new_pc
        self.frames.peek().pc = new_pc
        
        

class StateSet:
    per_inst : dict[PC, A]
    needswork : set[PC]

    def per_instruction(self):
        for pc in self.needswork: 
            yield (pc, self.per_inst[pc])

    def __init__(self, a: A, pc: PC):
        self.per_inst = {pc: a}
        self.needswork = {pc}

    def initialize(a: A, pc: PC) -> "StateSet":
        return StateSet(a, pc)

    # sts |= sts
    def __ior__(self, sts: Iterable[A]) -> "StateSet":
        logger.debug(f"MERGE {sts}")
        pc_temp  = set()
        for state in sts:
            logger.info(f"Merging state: {state}")
            if state.pc not in self.per_inst:
                self.per_inst[state.pc] = state
                pc_temp.add(state.pc)
            else:
                old = self.per_inst[state.pc]
                self.per_inst[state.pc] |= state
                # logger.debug(f"Merged state at {old != self.per_inst[state.pc]}")
                if old != self.per_inst[state.pc]:
                    pc_temp.add(state.pc)
        logger.debug(f"States needing work: {pc_temp}")
        self.needswork = pc_temp
        return self
  


def step(sts: StateSet ) -> Iterable[A | str]:
    states = copy.deepcopy(sts)
    for pc, state in states.per_instruction():
        s = copy.deepcopy(state)
        frame = s.frames.peek()
        offset = frame.pc.offset
        logger.info(f"STEP {pc} \n BC: {bc} \n STATE: {state} \n FRAME: {frame}")
        logger.info(f"STEP {bc[pc]}")
        
        match bc[pc]:
            case jvm.Get(field=field):
                # assert (field.extension.name == "$assertionsDisabled"), f"unknown field {field}"
                frame.stack.push(jvm.Value.boolean(False))
                frame.pc = PC(frame.pc.method, offset + 1)
                s.pc = frame.pc
                yield s

            case jvm.Push(value=v):
                frame.stack.push(v)
                frame.pc = PC(frame.pc.method, offset + 1)
                s.pc = frame.pc
                yield s

            case jvm.Store(type=t, index=i):
                v = frame.stack.pop()
                if isinstance(t, jvm.Int) or isinstance(t, jvm.Reference) or isinstance(t, jvm.Double):
                    frame.locals[i] = v
                elif isinstance(t, SignSet):
                    frame.locals[i] = v
                else:
                    raise NotImplementedError(f"Unhandled store type: {t}")
                frame.pc = PC(frame.pc.method, offset + 1)
                s.pc = frame.pc
                yield s
                    
            case jvm.Load(type=t, index=i):
                v = frame.locals[i]
                if isinstance(t, jvm.Int):
                    frame.stack.push(v)
                elif isinstance(t, jvm.Reference):
                    frame.stack.push(v)
                else:
                    raise NotImplementedError(f"Unhandled load type: {t}")
                frame.pc = PC(frame.pc.method, offset + 1)
                s.pc = frame.pc
                yield s

            case jvm.Goto(target=t):
                frame.pc = PC(frame.pc.method, t)
                yield state
            

            case jvm.Return(type=jvm.Int()):
                v1 = frame.stack.pop()
                s.frames.pop()
                if s.frames:
                    s.frames.peek().stack.push(v1)
                    yield s
                
            case jvm.Return(type=None):
                s.frames.pop()
                if s.frames:
                    yield s

            case jvm.Throw():
                break

            case jvm.Binary(operant=oper):
                v2, v1 = frame.stack.pop(), frame.stack.pop()
                # logger.debug(f"Binary operation {oper} on {v1} and {v2}, types {type(v1)}, {type(v2)}")
                if v1 is None or v2 is None:
                    break
                if isinstance(v1, jvm.Value | int):
                    v1: SignSet = SignSet.abstract_value(v1.value)
                if isinstance(v2, jvm.Value | int):
                    v2: SignSet = SignSet.abstract_value(v2.value)

                for s1 in v1.signs:
                    for s2 in v2.signs:
                        s1 = SignSet(s1)
                        s2 = SignSet({s2})
                        match oper:
                            case jvm.BinaryOpr.Div: 
                                if s2 == '0':
                                    break
                                elif s1 == '0':
                                    res = SignSet({'0'})
                                elif s1 == s2 :
                                    res = SignSet({'+'})
                                else:
                                    res = SignSet({'-'})
                            case jvm.BinaryOpr.Add:
                                if s1 == '0':
                                    res = s2
                                elif s2 == '0':
                                    res = s1
                                elif s1 == s2:
                                    res = s1
                                else:
                                    res = SignSet({'+', '-', '0'})
                            case jvm.BinaryOpr.Sub:
                                if s2 == '0':
                                    res = s1
                                elif s1 == '0':
                                    if s2 == '+':
                                        res = SignSet({'-'})
                                    elif s2 == '-':
                                        res = SignSet({ '+'})
                                elif s1 == s2:
                                    res = SignSet({'+', '-', '0'})
                                else: 
                                    res = s1
                            case jvm.BinaryOpr.Mul:
                                if s1 == '0' or s2 == '0':
                                    res = SignSet({'0'})
                                elif s1 == s2:
                                    res = SignSet({'+'})
                                else:
                                    res = SignSet({'-'})
                            case jvm.BinaryOpr.Rem:
                                if s2 == '0':
                                    break
                                elif s1 == '0':
                                    res = SignSet({'0'})
                                else:
                                    res = s1
                            case _:
                                raise NotImplementedError(f"Unhandled integer binary op: {oper}")
                        frame.stack.push(res) 
                        frame.pc = PC(frame.pc.method, offset + 1)
                        s.pc = frame.pc
                        yield s
                        s = copy.deepcopy(state)
                        frame = s.frames.peek()

      
            case jvm.Ifz(condition=cond, target=target):
                v = frame.stack.pop()
                logger.info(f"IFZ on value {v} with condition {cond}, and target {target}")

                if not isinstance(v, SignSet):
                    v: SignSet = SignSet.abstract_value(v)

                # logger.debug(f"IFZ on {v.signs} with condition {cond}")
                # logger.debug(f"Signs: {v}")

                temp_target = -1

                for sign in v.signs:
                    match cond:
                        case "eq":
                            if sign == "0" :
                                temp_target = target
                            else:
                                temp_target = offset + 1
                        case "ne":
                            if sign != "0":
                                temp_target = target
                            else:
                                temp_target = offset + 1
                        case "lt":
                            if sign == "-":
                                temp_target = target
                            else:
                                temp_target = offset + 1
                        case "gt":
                            if sign == "+":
                                temp_target = target
                            else:
                                temp_target = offset + 1
                        case "ge":
                            if sign == "+" or sign == "0":
                                temp_target = target
                            else:
                                temp_target = offset + 1
                        case "le":
                            if sign == "-" or sign == "0":
                                temp_target = target
                            else:
                                temp_target = offset + 1
                        case _:
                            raise NotImplementedError(f"Unhandled ifz condition: {cond}")
                    frame.pc = PC(frame.pc.method, temp_target)
                    s.pc = frame.pc
                    logger.info(f"TEMP_TARGET: {temp_target}")
                    yield s
                    s = copy.deepcopy(state)
                    frame = s.frames.peek()


            case _ : 
                logger.info(f"Unhandled opcode {bc[pc]}")
                frame.pc = frame.pc + 1
                s.pc = frame.pc
                yield s




suite = jpamb.Suite()
bc = Bytecode(suite, dict())
   
methodid, input = jpamb.getcase()
logger.info(f"Analyzing method {methodid.extension}\n {methodid} with input {input} and {methodid.methodid.params._elements}")


s = A.initialstate_from_method(methodid)
sts: StateSet = StateSet.initialize(s, PC(methodid, 0))

logger.info(f"Initial state setup {sts}")

final: set[str] = set()
MAX_STEPS = 10
for i in range(MAX_STEPS):
    new_states = step(sts)
    logger.info(f"After step {i}, new states: {new_states}")
    # for s in new_states:
    #     if isinstance(s, str):
    #         logger.info(f"Final state reached: {s}")
    #     else: 
    #         logger.info(f"Step {i}, program counter {s.pc.offset}")
    sts |= new_states
    if sts.needswork.__len__() == 0:
        logger.info(f"No more states to process after {i} steps.")
        break
    
    # for pc, st in sts.per_instruction():
        # logger.info(f"State at {pc.offset}: {st}")

for s in sts.per_inst:
    logger.info(f"All states at {s.offset}")

print("*")





