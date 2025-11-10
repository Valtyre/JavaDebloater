import jpamb
from jpamb import jvm
from dataclasses import dataclass

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
        
        case jvm.Load(type=jvm.Int(), index=i):
            frame.stack.push(frame.locals[i])
            frame.pc += 1
            return state
        
        case jvm.Binary(type=jvm.Int(), operant=operant):
            v2, v1 = frame.stack.pop(), frame.stack.pop()
            assert v1.type is jvm.Int(), f"expected int, but got {v1}"
            assert v2.type is jvm.Int(), f"expected int, but got {v2}"
            if v2.value == 0:
                return "divide by zero"
            
            match operant:
                case jvm.BinaryOpr.Div:
                    if v2.value == 0:
                        return "divide by zero"
                    else:
                        frame.stack.push(jvm.Value.int(v1.value // v2.value))

                case jvm.BinaryOpr.Mul:
                    frame.stack.push(jvm.Value.int(v1.value * v2.value))
                
                case jvm.BinaryOpr.Sub:
                    frame.stack.push(jvm.Value.int(v1.value - v2.value))

                case jvm.BinaryOpr.Add:
                    frame.stack.push(jvm.Value.int(v1.value + v2.value))

            frame.pc += 1
            return state
        
        case jvm.Get(field=field):
            assert (field.extension.name == "$assertionsDisabled"), f"unknown field {field}"
            frame.stack.push(jvm.Value.boolean(False))
            frame.pc += 1
            return state
        
        case jvm.Ifz(target=t, condition=c):
            assert c in ("ne", "gt", "lt", "eq", "ge", "le"), f"unknown condition {c}"
            assert isinstance(t, int), f"unknown target {t}"

            # if t <= frame.pc.offset:
            #     return "*"

            v1 = frame.stack.pop()
            print("IFZ:", v1, v1.value, c, t)
            match c:
                case "ne" if not v1.value:
                    frame.pc += 1
                case "gt" if v1.value > 0:
                    frame.pc += 1
                case "lt" if v1.value < 0:
                    frame.pc += 1
                case "eq" if v1.value:
                    frame.pc += 1
                case "ge" if v1.value >= 0:
                    frame.pc += 1
                case "le" if v1.value <= 0:
                    frame.pc += 1
                case _:
                    frame.pc = PC(frame.pc.method, t)
            print("IF: True", v1.value > 0, c == "gt")
            return state

        case jvm.If(target=t, condition=c):
            assert c in ("ne", "gt", "lt", "eq"), f"unknown condition {c}"
            assert isinstance(t, int), f"unknown target {t}"

            v1 = frame.stack.pop()
            print("IF:", v1, v1.value, c)
            match c:
                case "ne" if not v1.value:
                    frame.pc += 1
                case "gt" if v1.value > 0:
                    frame.pc += 1
                case "lt" if v1.value < 0:
                    frame.pc += 1
                case "eq" if v1.value:
                    frame.pc += 1
                case _:
                    frame.pc = PC(frame.pc.method, t)
            return state
        
        case jvm.New(classname=classname):
            assert isinstance(classname, jvm.ClassName), f"unknown class {classname}"
            print("NEW:", classname)
            frame.stack.push(classname)
            frame.pc += 1
            return state  

        case jvm.Dup():
            v1 = frame.stack.peek()
            frame.stack.push(v1)
            frame.pc += 1
            return state
        
        ## Loops

        case jvm.Goto(target=t):
            assert isinstance(t, int), f"unknown target {t}"
            frame.pc = PC(frame.pc.method, t)
            if frame.pc.offset == t:
                return "*"
            return state
        
        case jvm.Cast(from_=from_, to_=to_, ):
            v1 = frame.stack.pop()
            i = v1.value
            print(to_)

            frame.pc += 1

            match to_: 
                case jvm.Short():
                    if i < -32768 or i > 32767:
                        return "assertion error"
                    frame.stack.push(i)
                case _: 
                    raise NotImplementedError(f"Don't know how to cast to {t}")

            return state

        ## Calls

        case jvm.InvokeStatic(method=method):
            frame.pc += 1

            # Extract arguments from the stack
            num_params = len(method.extension.params)
            args = [frame.stack.pop() for _ in range(num_params)][::-1]
            print("INVOKE:", method, args)

            # Create a new frame for the invoked method
            new_frame = Frame.from_method(method)

            # Initialize the new frame's local variables with the arguments
            for i, arg in enumerate(args):
                new_frame.locals[i] = arg


            new_state = State({}, Stack.empty().push(new_frame))

            for x in range(100000):
                new_state = step(new_state)
                if isinstance(new_state, str):
                    print(new_state)
                    break
            else:
                print("*")
        ## Arrays 

        case jvm.NewArray(type=type, dim=dim):
            frame.pc += 1
            count = frame.stack.pop()
            print("NEWARRAY:", type, dim, count)
            assert count.type is jvm.Int(), f"expected int but got {count}"
            assert count.value >= 0, f"array size must be non-negative but got {count}"

            a = jvm.Value.array(type, ([] * count.value) * dim)
            frame.stack.push(jvm.Value.array(type, ([] * count.value) * dim))
            return state
        
        case jvm.Store(type=type, index=i):
            v1 = frame.stack.pop()
            frame.locals[i] = v1
            frame.pc += 1
            return state


        
        case jvm.InvokeSpecial(method=method):
            frame.pc += 1
            print("INVOKE:", method)
            if method.classname == "java/lang/AssertionError" and method.name == "<init>":
                return state
            else:
                frame.stack.pop()  # pop this
            return state
        
        case jvm.Throw():
            frame.pc += 1
            v1 = frame.stack.pop()
            assert isinstance(v1, jvm.ClassName), f"expected classname but got {v1}"   
            frame.stack.empty()
            print("THROW:", v1)
            if str(v1) == "java/lang/AssertionError":
                return "assertion error"
            return str(v1)

        case jvm.Return(type=jvm.Int()):
            v1 = frame.stack.pop()
            assert v1.type is jvm.Int(), f"expected int but got {v1}"

            state.frames.pop()

            if state.frames:
                frame = state.frames.peek()
                frame.stack.push(v1)
                frame.pc += 1
                return state
            else:
                return "ok"
        
        case jvm.Return():
                return "ok"
        

        case a:
            raise NotImplementedError(f"Don't know how to handle: {a!r}")
            sys.exit(f"Don't know how to handle: {a!r}")


frame = Frame.from_method(methodid)
for i, v in enumerate(input.values):
    print("INIT:", i, v)
    frame.locals[i] = v

state = State({}, Stack.empty().push(frame))

for x in range(100000):
    state = step(state)
    if isinstance(state, str):
        print(state)
        break
else:
    print("*")
