package jpamb.cases;

import jpamb.utils.*;
import static jpamb.utils.Tag.TagType.*;

public class Conditionals {

  @Case("(5) -> ok")
  @Case("(0) -> ok")
  @Case("(-5) -> ok")
  public static void ifElseChain(int x) {
    if (x > 0) {
      assert x > 0;
    } else if (x < 0) {
      assert x < 0;
    } else {
      assert x == 0;
    }
  }

  @Case("(true, true) -> ok")
  @Case("(true, false) -> assertion error")
  @Case("(false, true) -> ok")
  @Case("(false, false) -> ok")
  public static void andCondition(boolean a, boolean b) {
    if (a && b) {
      assert a;
      assert b;
    }
  }

  @Case("(true, true) -> ok")
  @Case("(true, false) -> ok")
  @Case("(false, true) -> ok")
  @Case("(false, false) -> assertion error")
  public static void orCondition(boolean a, boolean b) {
    if (a || b) {
      assert a || b;
    } else {
      assert false;
    }
  }

  @Case("(true) -> assertion error")
  @Case("(false) -> ok")
  public static void notCondition(boolean a) {
    if (!a) {
      assert !a;
    } else {
      assert false;
    }
  }

  @Case("(5, 10) -> ok")
  @Case("(10, 5) -> assertion error")
  public static void comparison(int a, int b) {
    if (a < b) {
      assert a < b;
    } else {
      assert false;
    }
  }

  @Case("(5) -> ok")
  @Case("(0) -> assertion error")
  @Case("(15) -> assertion error")
  public static void rangeCheck(int x) {
    if (x >= 1 && x <= 10) {
      assert x > 0 && x < 11;
    } else {
      assert false;
    }
  }

  @Case("(1) -> ok")
  @Case("(2) -> ok")
  @Case("(3) -> ok")
  @Case("(4) -> assertion error")
  public static void switchCase(int x) {
    switch (x) {
      case 1:
        assert x == 1;
        break;
      case 2:
        assert x == 2;
        break;
      case 3:
        assert x == 3;
        break;
      default:
        assert false;
    }
  }

  @Case("(10) -> ok")
  @Case("(5) -> assertion error")
  public static void switchWithFallthrough(int x) {
    switch (x) {
      case 10:
      case 20:
        assert x >= 10;
        break;
      default:
        assert false;
    }
  }

  @Case("(true, true, true) -> ok")
  @Case("(true, true, false) -> assertion error")
  public static void nestedConditions(boolean a, boolean b, boolean c) {
    if (a) {
      if (b) {
        if (c) {
          assert a && b && c;
        } else {
          assert false;
        }
      }
    }
  }

  @Case("(5, 3) -> ok")
  @Case("(3, 5) -> ok")
  public static int max(int a, int b) {
    if (a > b) {
      return a;
    } else {
      return b;
    }
  }

  @Case("(5, 3) -> ok")
  @Case("(3, 5) -> ok")
  public static int min(int a, int b) {
    if (a < b) {
      return a;
    } else {
      return b;
    }
  }

  @Case("(5, 3, 8) -> ok")
  public static void minMaxRange(int a, int b, int c) {
    int minimum = min(a, min(b, c));
    int maximum = max(a, max(b, c));
    assert minimum <= maximum;
  }

  @Case("(10) -> ok")
  @Case("(5) -> assertion error")
  public static void ternaryOperator(int x) {
    int result = (x > 5) ? x : 0;
    assert result > 5;
  }

  @Case("(0, 1) -> ok")
  @Case("(0, 0) -> divide by zero")
  public static void shortCircuitAnd(int a, int b) {
    if (b != 0 && a / b > 0) {
      assert true;
    }
  }

  @Case("(1, 0) -> ok")
  @Case("(0, 0) -> divide by zero")
  public static void noShortCircuitBitwise(int a, int b) {
    if (b != 0 & a / b > 0) {
      assert true;
    }
  }

  @Case("(5) -> ok")
  @Case("(-5) -> ok")
  @Case("(0) -> ok")
  public static void absoluteValue(int x) {
    int abs = (x < 0) ? -x : x;
    assert abs >= 0;
  }

  @Case("(100) -> ok")
  @Case("(50) -> assertion error")
  @Case("(0) -> assertion error")
  public static void gradeEvaluation(int score) {
    if (score >= 90) {
      assert score >= 90;
    } else if (score >= 80) {
      assert false;
    } else if (score >= 70) {
      assert false;
    } else {
      assert false;
    }
  }

  @Case("(2020) -> ok")
  @Case("(2021) -> assertion error")
  @Tag({ LOOP })
  public static void leapYear(int year) {
    boolean isLeap = false;
    if (year % 4 == 0) {
      if (year % 100 == 0) {
        if (year % 400 == 0) {
          isLeap = true;
        }
      } else {
        isLeap = true;
      }
    }
    assert isLeap;
  }

  @Case("(5, 5) -> ok")
  @Case("(5, 3) -> assertion error")
  public static void equalityCheck(int a, int b) {
    if (a == b) {
      assert a - b == 0;
    } else {
      assert false;
    }
  }
}
