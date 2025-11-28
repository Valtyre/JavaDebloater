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
class PerVarFrame:
    locals: dict[int, SignSet | jvm.Boolean]
    stack: Stack[SignSet]
    pc: PC

    def from_method(method: jvm.AbsMethodID) -> "PerVarFrame":
        return PerVarFrame({}, Stack.empty(), PC(method, 0))
    
    def __str__(self):
        locals = ", ".join(f"{k}:{v}" for k, v in self.locals.items())
        return f"<{{{locals}}}, {self.stack}, {self.pc}>"


@dataclass
class A: # Abstract State
    frames: Stack[PerVarFrame]
    pc: PC

    def initialstate_from_method(methodid: jvm.AbsMethodID) -> "A":
        frame: PerVarFrame = PerVarFrame.from_method(methodid)
        for i, v in enumerate(methodid.methodid.params._elements):
            if isinstance(v, jvm.Int):
                frame.locals[i] = SignSet.TOP()
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
        
        

class StateSet[A]:
    per_inst : dict[PC, A]
    needswork : set[PC]

    def per_instruction(self):
        for pc in self.needswork: 
            yield (pc, self.per_inst[pc])

    def __init__(self, a: A, pc: PC):
        self.per_inst = {pc: a}
        self.needswork = {pc}

    def initialize(a: A, pc: PC) -> "StateSet[A]":
        return StateSet(a, pc)

    # sts |= astate
    def __ior__(self, astate: A) -> "StateSet[A]":
        if astate.pc not in self.per_inst:
            self.per_inst[astate.pc] = astate
            self.needswork.add(astate.pc)
        else:
            old = self.per_inst[astate.pc]
            self.per_inst[astate.pc] |= astate
            if old != self.per_inst[astate.pc]:
                self.needswork.add(astate.pc)
        return self
  


def manystep(sts: StateSet[A] ) -> Iterable[A | str]:
    states = copy.deepcopy(sts)
    for pc, state in states.per_instruction():
        s = copy.deepcopy(state)
        frame = s.frames.peek()
        # logger.info(f"STEP {pc} \n BC: {bc} \n STATE: {state} \n FRAME: {frame}")
        logger.info(f"STEP {bc[pc]}")
        
        match bc[frame.pc]:
            case jvm.Get(field=field):
                # assert (field.extension.name == "$assertionsDisabled"), f"unknown field {field}"
                frame.stack.push(jvm.Value.boolean(False))
                frame.pc = frame.pc + 1
                s.pc = frame.pc
                yield state

            case jvm.Push(value=v):
                frame.stack.push(v)
                frame.pc = frame.pc + 1
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
                frame.pc = frame.pc + 1
                s.pc = frame.pc
                yield s

            case jvm.Return(type=jvm.Int()):
                v1 = frame.stack.pop()
                s.frames.pop()
                if s.frames:
                    s.frames.peek().stack.push(v1)
                    yield s
                else:
                    yield "ok"
                
            case jvm.Return(type=None):
                s.frames.pop()
                if s.frames:
                    yield s
                else:
                    yield "ok"

            case jvm.Binary(operant=oper):
                v2, v1 = frame.stack.pop(), frame.stack.pop()
                logger.debug(f"Binary operation {oper} on {v1} and {v2}, types {type(v1)}, {type(v2)}")
                if v1 is None or v2 is None:
                    break
                if isinstance(v1, jvm.Value | int):
                    v1: SignSet = SignSet.abstract(v1.value)
                if isinstance(v2, jvm.Value | int):
                    v2: SignSet = SignSet.abstract(v2.value)
                if oper == jvm.BinaryOpr.Div:
                    if '0' in v2.signs:
                        yield "divide by zero"
                        return
                    res = v1.div(v2)
                elif oper == jvm.BinaryOpr.Add:
                    res = v1.add(v2)
                elif oper == jvm.BinaryOpr.Sub:
                    res = v1.sub(v2)
                elif oper == jvm.BinaryOpr.Mul:
                    res = v1.mult(v2)
                elif oper == jvm.BinaryOpr.Rem:
                    if '0' in v2.signs:
                        yield "divide by zero"
                        return
                    res = v1.rem(v2)
                else:
                    raise NotImplementedError(f"Unhandled integer binary op: {oper}")
                frame.stack.push(res)
                frame.pc = frame.pc + 1
                s.pc = frame.pc
                yield s
      
            case jvm.Ifz(condition=cond, target=target):
                v = frame.stack.pop()

                # if not isinstance(v, SignSet):
                #     v: SignSet = SignSet.abstract(v)

                # logger.debug(f"IFZ on {v.signs} with condition {cond}")
                # logger.debug(f"Signs: {v}")

                for sign in v.signs:
                    match cond:
                        case "eq":
                            if sign == "0" :
                                frame.pc = PC(frame.pc.method, target)
                            else:
                                frame.pc = frame.pc + 1
                        case "ne":
                            if sign != "0":
                                frame.pc = PC(frame.pc.method, target) 
                            else:
                                frame.pc = frame.pc + 1
                        case "lt":
                            if sign == "-":
                                frame.pc = PC(frame.pc.method, target)
                            else:  
                                frame.pc = frame.pc + 1
                        case "gt":
                            if sign == "+":
                                frame.pc = PC(frame.pc.method, target)
                            else:
                                frame.pc = frame.pc + 1
                        case "ge":
                            if sign == "+" or sign == "0":
                                frame.pc = PC(frame.pc.method, target)
                            else:
                                frame.pc = frame.pc + 1
                        case "le":
                            if sign == "-" or sign == "0":
                                frame.pc = PC(frame.pc.method, target)
                            else:
                                frame.pc = frame.pc + 1
                        case _:
                            raise NotImplementedError(f"Unhandled ifz condition: {cond}")
                    s.pc = frame.pc
                    yield s


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
sts: StateSet[A] = StateSet[A].initialize(s, PC(methodid, 0))

logger.info(f"Initial state setup {sts}")

final: set[str] = set()
MAX_STEPS = 5
for i in range(MAX_STEPS):
  for s in manystep(sts):
    if isinstance(s, A):
        logger.info(f"Step {i}, program counter {s.pc.offset}")
    if isinstance(s, str):
      logger.info(f"Final state reached: {s}")
      final.add(s)
    else:
      sts |= s

if len(final) == 0:
    print("*")
else: 
    for f in final:
        print(f)




