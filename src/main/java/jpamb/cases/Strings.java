package jpamb.cases;

import jpamb.utils.*;
import static jpamb.utils.Tag.TagType.*;

public class Strings {

  @Case("() -> ok")
  public static void emptyStringLength() {
    String s = "";
    assert s.length() == 0;
  }

  @Case("() -> ok")
  public static void stringLength() {
    String s = "hello";
    assert s.length() == 5;
  }

  @Case("(\"abc\", \"def\") -> ok")
  @Case("(\"\", \"\") -> ok")
  public static void concatenation(String a, String b) {
    String result = a + b;
    assert result.length() >= a.length();
    assert result.length() >= b.length();
  }

  @Case("(\"hello\", 0) -> ok")
  @Case("(\"hello\", 5) -> out of bounds")
  @Case("(\"hello\", -1) -> out of bounds")
  public static void charAt(String s, int index) {
    char c = s.charAt(index);
    assert c >= 0;
  }

  @Case("(\"hello\", \"h\") -> ok")
  @Case("(\"hello\", \"x\") -> assertion error")
  public static void startsWith(String s, String prefix) {
    assert s.startsWith(prefix);
  }

  @Case("(\"hello\", \"o\") -> ok")
  @Case("(\"hello\", \"x\") -> assertion error")
  public static void endsWith(String s, String suffix) {
    assert s.endsWith(suffix);
  }

  @Case("(\"hello world\", \" \") -> ok")
  public static void split(String s, String delimiter) {
    String[] parts = s.split(delimiter);
    assert parts.length > 0;
  }

  @Case("(\"  trim  \") -> ok")
  public static void trim(String s) {
    String trimmed = s.trim();
    assert trimmed.length() <= s.length();
  }

  @Case("(\"HELLO\") -> ok")
  @Case("(\"hello\") -> ok")
  public static void toLowerCase(String s) {
    String lower = s.toLowerCase();
    assert lower.length() == s.length();
  }

  @Case("(\"hello\") -> ok")
  @Case("(\"HELLO\") -> ok")
  public static void toUpperCase(String s) {
    String upper = s.toUpperCase();
    assert upper.length() == s.length();
  }

  @Case("(\"hello\", \"ell\") -> ok")
  @Case("(\"hello\", \"xyz\") -> assertion error")
  public static void contains(String s, String substring) {
    assert s.contains(substring);
  }

  @Case("(\"hello\", \"l\") -> ok")
  public static void indexOf(String s, String target) {
    int index = s.indexOf(target);
    assert index >= 0;
  }

  @Case("(\"hello\", \"x\") -> assertion error")
  public static void indexOfNotFound(String s, String target) {
    int index = s.indexOf(target);
    assert index >= 0;
  }

  @Case("(\"hello\", 1, 4) -> ok")
  @Case("(\"hello\", 0, 10) -> out of bounds")
  public static void substring(String s, int start, int end) {
    String sub = s.substring(start, end);
    assert sub.length() >= 0;
  }

  @Case("(\"hello\", \"hello\") -> ok")
  @Case("(\"hello\", \"world\") -> assertion error")
  public static void equals(String a, String b) {
    assert a.equals(b);
  }

  @Case("(\"Hello\", \"hello\") -> ok")
  @Case("(\"hello\", \"world\") -> assertion error")
  public static void equalsIgnoreCase(String a, String b) {
    assert a.equalsIgnoreCase(b);
  }

  @Case("(\"abc\") -> ok")
  public static void stringBuilder() {
    StringBuilder sb = new StringBuilder();
    sb.append("a");
    sb.append("b");
    sb.append("c");
    String result = sb.toString();
    assert result.equals("abc");
  }

  @Case("(\"hello\", 'l', 'r') -> ok")
  public static void replace(String s, char oldChar, char newChar) {
    String replaced = s.replace(oldChar, newChar);
    assert replaced.length() == s.length();
  }

  @Case("() -> ok")
  public static void stringComparison() {
    String a = "test";
    String b = "test";
    assert a.compareTo(b) == 0;
  }

  @Case("(\"abc\", \"xyz\") -> assertion error")
  @Case("(\"abc\", \"abc\") -> ok")
  public static void compareStrings(String a, String b) {
    assert a.compareTo(b) <= 0;
  }
}
