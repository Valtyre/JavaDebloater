from dataclasses import dataclass
from typing import TypeAlias, Literal

from dataclasses import dataclass
from typing import TypeAlias, Literal
from jpamb import jvm

# Define Sign as a type alias for the allowed literals
Sign: TypeAlias = Literal["+", "-", "0"]

@dataclass
class SignSet:
    signs: set[Sign]

    def __ior__(self, other: "SignSet") -> "SignSet":
        self.signs |= other.signs
        return self

    def sing_to_signset(sign: Sign) -> "SignSet":
        return SignSet({sign})

    @classmethod
    def bottom(cls) -> "SignSet":
        return cls(set())

    @classmethod
    def top(cls) -> "SignSet":
        return cls({"+", "-", "0"})

    @classmethod
    def pos(cls) -> "SignSet":
        return cls({"+",})

    @classmethod
    def neg(cls) -> "SignSet":
        return cls({"-",})

    @classmethod
    def zero(cls) -> "SignSet":
        return cls({"0",})

    def is_bottom(self) -> bool:
        return len(self.signs) == 0

    def is_top(self) -> bool:
        return self.signs == {"+", "-", "0"}

    def join(self, other: "SignSet") -> "SignSet":
        return SignSet(self.signs | other.signs)

    def meet(self, other: "SignSet") -> "SignSet":
        return SignSet(self.signs & other.signs)

    def contains(self, sign: str) -> bool:
        return sign in self.signs


    @classmethod
    def from_int(cls, n: int) -> "SignSet":
        if n < 0:
            return cls(frozenset({'-'}))
        if n == 0:
            return cls(frozenset({'0'}))
        return cls(frozenset({'+'}))

    @classmethod
    def from_float(cls, x: float) -> "SignSet":
        if x < 0.0:
            return cls(frozenset({'-'}))
        if x == 0.0:
            return cls(frozenset({'0'}))
        return cls(frozenset({'+'}))

    @classmethod
    def abstract_value(cls, v) -> "SignSet":
        
        if isinstance(v, SignSet):
            return v

        if isinstance(v, bool):
            return cls.top()

        if isinstance(v, int):
            return cls.from_int(v)

        if isinstance(v, float):
            return cls.from_float(v)

        return cls.bottom()
    
    def add(self, other: "SignSet") -> "SignSet":
        result_signs = set()
        if self.contains("+") and other.contains("+"):
            result_signs.add("+")
        if "-" in self.signs and "-" in other.signs:
            result_signs.add("-")
        if ("+" in self.signs and "-" in other.signs) or ("-" in self.signs and "+" in other.signs):
            result_signs.add("0")
        if "0" in self.signs:
            result_signs.update(other.signs)
        if "0" in other.signs:
            result_signs.update(self.signs)
        return SignSet(result_signs)
    
    def sub(self, other: "SignSet") -> "SignSet":
        result_signs = set()
        if "+" in self.signs and "-" in other.signs:
            result_signs.add("+")
        if "-" in self.signs and "+" in other.signs:
            result_signs.add("-")
        if ("+" in self.signs and "+" in other.signs) or ("-" in self.signs and "-" in other.signs):
            result_signs.add("0")
        if "0" in self.signs:
            for s in other.signs:
                if s == "+":
                    result_signs.add("-")
                elif s == "-":
                    result_signs.add("+")
                else:
                    result_signs.add("0")
        if "0" in other.signs:
            result_signs.update(self.signs)
        return SignSet(result_signs)
    
    def mult(self, other: "SignSet") -> "SignSet":
        result_signs = set()
        for s1 in self.signs:
            for s2 in other.signs:
                if s1 == "0" or s2 == "0":
                    result_signs.add("0")
                elif s1 == s2:
                    result_signs.add("+")
                else:
                    result_signs.add("-")
        return SignSet(result_signs)
    
    def div(self, other: "SignSet") -> "SignSet":
        result_signs = set()
        for s1 in self.signs:
            for s2 in other.signs:
                if s2 == "0":
                    continue
                if s1 == "0":
                    result_signs.add("0")
                elif s1 == s2:
                    result_signs.add("+")
                else:
                    result_signs.add("-")
        return SignSet(result_signs)
    
    def rem(self, other: "SignSet") -> "SignSet":
        result_signs = set()
        if "+" in self.signs:
            result_signs.add("+")
            result_signs.add("0")
        if "-" in self.signs:
            result_signs.add("-")
            result_signs.add("0")
        return SignSet(result_signs)