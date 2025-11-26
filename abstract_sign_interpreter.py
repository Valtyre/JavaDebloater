import jpamb
from jpamb import jvm
from dataclasses import dataclass
import copy
import math

import sign

import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="[{level}] {message}")

methodid, input = jpamb.getcase()

### Work here 

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

def bind_args_to_locals(frame, args):
    local_index = 0
    for arg in args:
        frame.locals[local_index] = arg
        if isinstance(arg.type, (jvm.Long, jvm.Double)):
            local_index += 2
        else:
            local_index += 1

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
    match opr:
        case jvm.Push(value=v):
            frame.stack.push(v)
            frame.pc += 1
            return state
        
        case jvm.Load(type=t, index=i):
            v = frame.locals[i]
            if isinstance(t, jvm.Int) or isinstance(t, jvm.Reference) or isinstance(t, jvm.Double):
                frame.stack.push(v)
            else:
                raise NotImplementedError(f"Unhandled load type: {t}")
            frame.pc += 1
            return state

        case jvm.ArrayLoad():
            index = frame.stack.pop().value
            arr = frame.stack.pop()
            if not isinstance(arr, list):
                raise RuntimeError(f"Expected array, got {arr}")
            if isinstance(index, sign.SignSet):
                frame.pc += 1
                return state
            elif index < 0 or index >= len(arr):
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
        
        ### Integer operations

        case jvm.Incr(index=i, amount=amt):
            v = frame.locals[i]
            if not isinstance(v, sign.SignSet):
                v: sign.SignSet = sign.SignSet.abstract_value( v.value)
            if not isinstance(amt, sign.SignSet):
                amt: sign.SignSet = sign.SignSet.abstract_value( amt)
            
            v = v.add(sign.SignSet.abstract_value( amt))
            frame.pc += 1
            return state
        
        case jvm.Binary(type=jvm.Int() | sign.SignSet | jvm.Double(), operant=oper):
            v2, v1 = frame.stack.pop(), frame.stack.pop()

            if not isinstance(v1, sign.SignSet):
                v1: sign.SignSet = sign.SignSet.abstract_value( v1.value)
            if not isinstance(v2, sign.SignSet):
                v2: sign.SignSet = sign.SignSet.abstract_value( v2.value)

            if oper == jvm.BinaryOpr.Div:
                res = v1.div(v2)
            elif oper == jvm.BinaryOpr.Add:
                res = v1.add(v2)
            elif oper == jvm.BinaryOpr.Sub:
                res = v1.sub(v2)
            elif oper == jvm.BinaryOpr.Mul:
                res = v1.mult(v2)
            elif oper == jvm.BinaryOpr.Rem:
                res = v1.rem(v2)
            else:
                raise NotImplementedError(f"Unhandled integer binary op: {oper}")

            frame.stack.push(res)
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
        
        case jvm.InvokeDynamic(offset=offset, stack_size=stack_size):
            args = []
            for _ in range(stack_size):
                args.append(frame.stack.pop())

            args.reverse()
        
            frame.stack.push(jvm.Value(jvm.String(), "<dyn-string>"))

            frame.pc += 1
            return state
        
        case jvm.InvokeVirtual(method=m):
            cname = m.classname.dotted()
            mname = m.methodid.name

            if (cname in ("java/lang/String", "java.lang.String")
                    and mname == "length"):
                frame.stack.pop()
                frame.stack.push(jvm.Value.int(1)) 

                frame.pc += 1
                return state

            arg_count = len(m.methodid.params._elements) + 1

            args: list[jvm.Value] = []
            for _ in range(arg_count):
                args.append(frame.stack.pop())
            args.reverse()

            newframe = Frame.from_method(m)
            for i, arg in enumerate(args):
                newframe.locals[i] = arg

            state.frames.push(newframe)
            frame.pc += 1
            return state
        
        

        case jvm.CompareFloating(type=t, mode=mode):
            v2 = frame.stack.pop()
            v1 = frame.stack.pop()

            def as_number(v):
                if isinstance(v, jvm.Value):
                    if isinstance(
                        v.type,
                        (
                            jvm.Double,
                            jvm.Float,
                            jvm.Int,
                            jvm.Long,
                            jvm.Short,
                            jvm.Byte,
                        ),
                    ):
                        return v.value
                    return None

                if v.__class__.__name__ == "SignSet":
                    return 0.0

                if isinstance(v, (int, float)):
                    return v
                return None

            x = as_number(v1)
            y = as_number(v2)

            
            if x is None or y is None:
                frame.stack.push(jvm.Value.int(0))
                frame.pc += 1
                return state

            import math

            m = (mode or "").lower()
            is_less_variant = m in ("l", "lt", "less", "cmpl")
            is_great_variant = m in ("g", "gt", "greater", "cmpg")

            nan = (
                isinstance(x, float) and math.isnan(x)
            ) or (
                isinstance(y, float) and math.isnan(y)
            )

            if nan:
                if is_less_variant:
                    res = -1
                else:
                    res = 1
            else:
                if x < y:
                    res = -1
                elif x > y:
                    res = 1
                else:
                    res = 0

            frame.stack.push(jvm.Value.int(res))
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
            if isinstance(t, jvm.Int) or isinstance(t, jvm.Reference) or isinstance(t, jvm.Double):
                frame.locals[i] = v
            elif isinstance(t, sign.SignSet):
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

            if not isinstance(v, sign.SignSet):
                v: sign.SignSet = sign.SignSet.abstract_value(v.value)

            take_branch = False
            if cond == "eq":
                take_branch = (v.contains("0"))
            elif cond == "ne":
                take_branch = (not v.contains("0"))
            elif cond == "lt":
                take_branch = (v.contains("-"))
            elif cond == "gt":
                take_branch = (v.contains("+"))
            elif cond == "ge":
                take_branch = (v.contains("0") or v.contains("+"))
            elif cond == "le":
                take_branch = (v.contains("0") or v.contains("-"))
            else:
                raise NotImplementedError(f"Unhandled ifz condition: {cond}")

            if take_branch:
                frame.pc = PC(frame.pc.method, target)
            else:
                frame.pc += 1
            return state

        case jvm.If(condition=cond, target=target):
            # Pop right, then left (same order as your concrete interpreter)
            v2 = frame.stack.pop()
            v1 = frame.stack.pop()

            # --- Normalise both to SignSet ---

            if not isinstance(v1, sign.SignSet):
                v1 = sign.SignSet.abstract_value(v1.value)
                
            if not isinstance(v2, sign.SignSet):
                v2 = sign.SignSet.abstract_value(v2.value)

            def has(s: sign.SignSet, sym: str) -> bool:
                return sym in s.signs


            if cond == "eq":
                take_branch = not v1.signs.isdisjoint(v2.signs)

            elif cond == "ne":
                take_branch = v1.signs != v2.signs

            elif cond == "lt":
                take_branch = has(v1, "-") and (has(v2, "0") or has(v2, "+"))

            elif cond == "le":
                take_branch = has(v1, "-") or has(v1, "0")

            elif cond == "ge":
                take_branch = has(v1, "+") or has(v1, "0")

            elif cond == "gt":
                if v2.signs == {"-"}:
                    take_branch = True
                else:
                    take_branch = has(v1, "+") and not has(v2, "+")
            else:
                raise NotImplementedError(f"Unhandled If condition: {cond}")
            
            print(take_branch)
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
bind_args_to_locals(frame, input.values)

state = State({}, Stack.empty().push(frame))

for x in range(1000):
    state = step(state)
    print(state)
    if isinstance(state, str):
        print(state)
        break
    else: 
        print("STATE: ",state)
else:
    print("*")