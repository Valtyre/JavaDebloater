from dataclasses import dataclass
from typing import TypeAlias, Literal

Sign : TypeAlias = Literal["+"] | Literal["-"] | Literal["0"]

@dataclass
class SignSet:
    signs : set[Sign]

    def __contains__(self, member : int): 
        if (member == 0 and "0" in self.signs): 
            return True
        elif (member > 0 and "+" in self.signs): 
            return True
        elif (member < 0 and "-" in self.signs):
            return True
        else:
            return False

    def abstract(cls, items : set[int]): 
        signset = set()
        if 0 in items:
          signset.add("0")
        if items.__contains__
            signset.add("+")
        if any(x < 0 for x in items):
            signset.add("-")
        return cls(signset)





from hypothesis import given
from hypothesis.strategies import integers, sets

@given(sets(integers()))
def test_valid_abstraction(xs: set[int]):
  s = SignSet.abstract(xs) 
  assert all(x in s for x in xs)