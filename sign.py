from dataclasses import dataclass
from typing import TypeAlias, Literal

Sign : TypeAlias = Literal["+"] | Literal["-"] | Literal["0"]

@dataclass
class SignSet:
    signs: set[Sign]

    def __contains__(self, member: str):
        if member in self.signs:
            return True
        return False

    @classmethod
    def abstract(self, cls, num: int):
        signset = set()
        if num == 0:
            signset.add("0")
        if num > 0:
            signset.add("+")
        if num < 0: 
            signset.add("-")
        return cls(signset)
    
    def literal(self): 
        return self.signs
    
    def add(self, other: "SignSet") -> "SignSet":
        result_signs = set()
        if "+" in self.signs and "+" in other.signs:
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