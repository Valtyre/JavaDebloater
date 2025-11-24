from dataclasses import dataclass
from typing import Literal, TypeAlias

FloatSign: TypeAlias = Literal["+", "-", "0"]

@dataclass
class DoubleAbs:
    signs: set[FloatSign]

    @classmethod
    def top(cls) -> "DoubleAbs":
        return cls({"+", "-", "0"})

    @classmethod
    def abstract(cls, x: float) -> "DoubleAbs":
        if x == 0.0:
            return cls({"0"})
        elif x > 0.0:
            return cls({"+"})
        else:
            return cls({"-"})

    def contains(self, s: FloatSign) -> bool:
        return s in self.signs

    def add(self, other: "DoubleAbs") -> "DoubleAbs":
        res: set[FloatSign] = set()

        if self.contains("+") and other.contains("+"):
            res.add("+")

        if self.contains("-") and other.contains("-"):
            res.add("-")

        if (self.contains("+") and other.contains("-")) or (self.contains("-") and other.contains("+")):
            res.add("0")

        if self.contains("0"):
            res |= other.signs
        if other.contains("0"):
            res |= self.signs

        if not res:
            res = {"+", "-", "0"}
        return DoubleAbs(res)

    def sub(self, other: "DoubleAbs") -> "DoubleAbs":
        flipped = DoubleAbs(
            { "+" if s == "-" else "-" if s == "+" else "0" for s in other.signs }
        )
        return self.add(flipped)

    def mul(self, other: "DoubleAbs") -> "DoubleAbs":
        res: set[FloatSign] = set()
        for s1 in self.signs:
            for s2 in other.signs:
                if s1 == "0" or s2 == "0":
                    res.add("0")
                elif s1 == s2:
                    res.add("+")
                else:
                    res.add("-")
        if not res:
            res = {"+", "-", "0"}
        return DoubleAbs(res)

    def div(self, other: "DoubleAbs") -> "DoubleAbs":
        res: set[FloatSign] = set()
        for s1 in self.signs:
            for s2 in other.signs:
                if s2 == "0":
                    continue
                if s1 == "0":
                    res.add("0")
                elif s1 == s2:
                    res.add("+")
                else:
                    res.add("-")
        if not res:
            res = {"+", "-", "0"}
        return DoubleAbs(res)


@dataclass
class AbsString:
    can_be_null: bool
    can_be_nonnull: bool

    @classmethod
    def null(cls) -> "AbsString":
        return cls(can_be_null=True, can_be_nonnull=False)

    @classmethod
    def nonnull(cls) -> "AbsString":
        return cls(can_be_null=False, can_be_nonnull=True)

    @classmethod
    def unknown(cls) -> "AbsString":
        return cls(can_be_null=True, can_be_nonnull=True)

    def join(self, other: "AbsString") -> "AbsString":
        return AbsString(
            can_be_null=self.can_be_null or other.can_be_null,
            can_be_nonnull=self.can_be_nonnull or other.can_be_nonnull,
        )