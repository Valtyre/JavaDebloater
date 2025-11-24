import jpamb
from jpamb import jvm
from dataclasses import dataclass
import copy

import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="[{level}] {message}")

methodid, input = jpamb.getcase()



@dataclass
class PC:
    method: jvm.AbsMethodID
    offset: int

    def __iadd__(self, delta):
        self.offset += delta
        return self

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


suite = jpamb.Suite()
bc = Bytecode(suite, dict())


@dataclass
class Frame:
    locals: dict[int, jvm.Value]
    stack: Stack[jvm.Value]
    pc: PC

    def __str__(self):
        locals = ", ".join(f"{k}:{v}" for k, v in sorted(self.locals.items()))
        return f"<{{{locals}}}, {self.stack}, {self.pc}>"

    def from_method(method: jvm.AbsMethodID) -> "Frame":
        return Frame({}, Stack.empty(), PC(method, 0))


@dataclass
class State:
    heap: dict[int, jvm.Value]
    frames: Stack[Frame]

    def __str__(self):
        return f"{self.heap} {self.frames}"


def step(state: State) -> State | str:
    assert isinstance(state, State), f"expected frame but got {state}"
    frame = state.frames.peek()
    opr = bc[frame.pc]
    logger.debug(f"STEP {opr}\n{state}")
    match opr:
        case jvm.Push(value=v):
            frame.stack.push(v)
            frame.pc += 1
            return state
        
        case jvm.Load(type=t, index=i):
            v = frame.locals[i]
            if isinstance(t, jvm.Int) or isinstance(t, jvm.Reference):
                frame.stack.push(v)
            else:
                raise NotImplementedError(f"Unhandled load type: {t}")
            frame.pc += 1
            return state

        case jvm.ArrayLoad(type=jvm.Int()):
            index = frame.stack.pop().value
            arr = frame.stack.pop()
            if not isinstance(arr, list):
                raise RuntimeError(f"Expected array, got {arr}")
            if index < 0 or index >= len(arr):
                return "out of bounds"
            frame.stack.push(jvm.Value.int(arr[index]))
            frame.pc += 1
            return state

        case jvm.ArrayStore(type=jvm.Int()):
            val = frame.stack.pop()
            index = frame.stack.pop().value
            arr = frame.stack.pop()
            if not isinstance(arr, list):
                raise RuntimeError(f"Expected array, got {arr}")
            if index < 0 or index >= len(arr):
                return "out of bounds"
            arr[index] = val.value
            frame.pc += 1
            return state
        
        case jvm.ArrayLength():
            arr = frame.stack.pop()
            if not isinstance(arr, list):
                raise RuntimeError(f"Expected array, got {arr}")
            frame.stack.push(jvm.Value.int(len(arr)))
            frame.pc += 1
            return state
        
        case jvm.Incr(index=i, amount=amt):
            v = frame.locals[i]
            assert v.type is jvm.Int(), f"expected int in local {i}, got {v}"
            frame.locals[i] = jvm.Value.int(v.value + amt)
            frame.pc += 1
            return state

        
        
        case jvm.Binary(type=jvm.Int(), operant=oper):
            v2, v1 = frame.stack.pop(), frame.stack.pop()
            assert v1.type is jvm.Int(), f"expected int, but got {v1}"
            assert v2.type is jvm.Int(), f"expected int, but got {v2}"
            if oper == jvm.BinaryOpr.Div:
                if v2.value == 0:
                    return "divide by zero"
                res = v1.value // v2.value
            elif oper == jvm.BinaryOpr.Add:
                res = v1.value + v2.value
            elif oper == jvm.BinaryOpr.Sub:
                res = v1.value - v2.value
            elif oper == jvm.BinaryOpr.Mul:
                res = v1.value * v2.value
            elif oper == jvm.BinaryOpr.Rem:
                if v2.value == 0:
                    return "divide by zero"
                res = v1.value % v2.value
            else:
                raise NotImplementedError(f"Unhandled integer binary op: {oper}")

            frame.stack.push(jvm.Value.int(res))
            frame.pc += 1
            return state
        case jvm.Return(type=jvm.Int()):
            v1 = frame.stack.pop()
            state.frames.pop()
            if state.frames:
                state.frames.peek().stack.push(v1)
                return state
            else:
                return "ok"
        
        case jvm.Return(type=None):
            state.frames.pop()
            if state.frames:
                return state
            else:
                return "ok"
        case jvm.New(classname=classname):
            frame.stack.push(classname)
            frame.pc += 1
            return state 
        
        case jvm.InvokeSpecial(method=method):
            frame.pc += 1
            return state       
        
        case jvm.InvokeStatic(method=method):
            arg_count = len(method.methodid.params._elements)
            args = []
            for _ in range(arg_count):
                args.append(frame.stack.pop())
            args.reverse()

            newframe = Frame.from_method(method)
            for i, arg in enumerate(args):
                newframe.locals[i] = arg

            state.frames.push(newframe)
            frame.pc += 1

            return state
        
        case jvm.Throw():
            v1 = frame.stack.pop()
            if str(v1) == "java/lang/AssertionError":
                return "assertion error"
            return str(v1)

        case jvm.Dup():
            if not frame.stack:
                raise RuntimeError("stack underflow on dup")
            v = frame.stack.peek()
            cv = copy.copy(v)
            frame.stack.push(cv)
            frame.pc += 1
            return state
        case jvm.Store(type=t, index=i):
            v = frame.stack.pop()
            if isinstance(t, jvm.Int) or isinstance(t, jvm.Reference):
                frame.locals[i] = v
            else:
                raise NotImplementedError(f"Unhandled store type: {t}")
            frame.pc += 1
            return state
   
        case jvm.Get(field=field):
            assert (field.extension.name == "$assertionsDisabled"), f"unknown field {field}"
            frame.stack.push(jvm.Value.boolean(False))
            frame.pc += 1
            return state
        case jvm.Ifz(condition=cond, target=target):
            v = frame.stack.pop()
            assert v.type in (jvm.Int(), jvm.Boolean()), f"expected int/bool, got {v}"

            take_branch = False
            if cond == "eq":
                take_branch = (v.value == 0)
            elif cond == "ne":
                take_branch = (v.value != 0)
            elif cond == "lt":
                take_branch = (v.value < 0)
            elif cond == "ge":
                take_branch = (v.value >= 0)
            elif cond == "gt":
                take_branch = (v.value > 0)
            elif cond == "le":
                take_branch = (v.value <= 0)
            else:
                raise NotImplementedError(f"Unhandled ifz condition: {cond}")

            if take_branch:
                frame.pc = PC(frame.pc.method, target)
            else:
                frame.pc += 1
            return state

        case jvm.If(condition=cond, target=target):
            v2 = frame.stack.pop()
            v1 = frame.stack.pop()
            assert v1.type is jvm.Int() and v2.type is jvm.Int()

            take_branch = False
            if cond == "eq":
                take_branch = (v1.value == v2.value)
            elif cond == "ne":
                take_branch = (v1.value != v2.value)
            elif cond == "lt":
                take_branch = (v1.value < v2.value)
            elif cond == "ge":
                take_branch = (v1.value >= v2.value)
            elif cond == "gt":
                take_branch = (v1.value > v2.value)
            elif cond == "le":
                take_branch = (v1.value <= v2.value)
            else:
                raise NotImplementedError(f"Unhandled If condition: {cond}")

            if take_branch:
                frame.pc = PC(frame.pc.method, target)
            else:
                frame.pc += 1
            return state
        
        case jvm.Goto(target=t):
            assert isinstance(t, int), f"unknown target {t}"
            frame.pc = PC(frame.pc.method, t)
            return state
        
        case jvm.Cast(from_=from_, to_ = to_):
            v1 = frame.stack.pop()
            i = v1.value
            frame.pc += 1
            match to_:
                case jvm.Short():
                    frame.stack.push(i)
                case _:
                    raise NotImplementedError(f"Unhandled If condition: {cond}")
            return state
        
        case jvm.NewArray(type=t, dim=1):
            size = frame.stack.pop().value
            if size < 0:
                return "out of bounds"
            arr = [0] * size
            frame.stack.push(arr)
            frame.pc += 1
            return state
        
        case a:
            raise NotImplementedError(f"Don't know how to handle: {a!r}")


frame = Frame.from_method(methodid)
for i, v in enumerate(input.values):
    frame.locals[i] = v

state = State({}, Stack.empty().push(frame))

for x in range(1000):
    state = step(state)
    if isinstance(state, str):
        print(state)
        break
    else: 
        print("STATE: ",state)
else:
    print("*")
