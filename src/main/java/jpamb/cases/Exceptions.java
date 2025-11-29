package jpamb.cases;

import jpamb.utils.*;
import static jpamb.utils.Tag.TagType.*;

public class Exceptions {

  @Case("() -> null pointer")
  public static void dereferenceNull() {
    Object obj = null;
    obj.toString();
  }

  @Case("() -> ok")
  public static void checkBeforeDeref() {
    Object obj = null;
    if (obj != null) {
      obj.toString();
    }
  }

  @Case("(true) -> ok")
  @Case("(false) -> null pointer")
  public static void conditionalNull(boolean createObject) {
    Object obj = null;
    if (createObject) {
      obj = new Object();
    }
    obj.toString();
  }

  @Case("() -> null pointer")
  public static void nullString() {
    String str = null;
    assert str.length() > 0;
  }

  @Case("() -> ok")
  public static void emptyString() {
    String str = "";
    assert str.length() == 0;
  }

  @Case("(5) -> ok")
  @Case("(0) -> null pointer")
  public static void sometimesNull(int x) {
    String result = null;
    if (x > 0) {
      result = "positive";
    }
    assert result.length() > 0;
  }

  @Case("() -> class cast")
  public static void badCast() {
    Object obj = "string";
    Integer num = (Integer) obj;
  }

  @Case("() -> ok")
  public static void goodCast() {
    Object obj = "string";
    String str = (String) obj;
    assert str.length() >= 0;
  }

  @Case("(10) -> ok")
  @Case("(-5) -> negative array size")
  public static void negativeArraySize(int size) {
    int[] arr = new int[size];
    assert arr.length >= 0;
  }

  @Case("() -> ok")
  public static void catchException() {
    try {
      int result = 1 / 0;
      assert false;  // Should not reach here
    } catch (ArithmeticException e) {
      assert true;   // Expected path
    }
  }

  @Case("() -> divide by zero")
  public static void uncaughtException() {
    try {
      int result = 1 / 0;
    } catch (NullPointerException e) {
      // Wrong exception type
    }
  }

  @Case("(null) -> null pointer")
  @Case("(\"test\") -> ok")
  public static void nullCheck(String input) {
    assert input.length() >= 0;
  }

  @Case("() -> ok")
  public static void multipleNullChecks() {
    Object a = null;
    Object b = null;
    if (a == null && b == null) {
      return;
    }
    a.toString();
  }

  @Case("() -> null pointer")
  public static void nullInArray() {
    String[] arr = new String[5];
    assert arr[0].length() > 0;
  }

  @Case("() -> ok")
  public static void initializedArray() {
    String[] arr = new String[] { "a", "b", "c" };
    assert arr[0].length() > 0;
  }
}
