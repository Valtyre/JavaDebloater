package jpamb.cases;

import jpamb.utils.Case;

public class Signs {

    @Case("() -> ok")
    public static int returnPositive() {
        return 1;
    }

    @Case("() -> ok")
    public static int returnZero() {
        return 0;
    }

    @Case("() -> ok")
    public static int returnNegative() {
        return -5;
    }

    
    @Case("(5) -> ok")
    @Case("(0) -> ok")
    @Case("(-5) -> ok")
    public static void classifySign(int x) {
        if (x > 0) {
            assert x > 0;   // positive branch
        } else if (x == 0) {
            assert x == 0;  // zero branch
        } else {
            assert x < 0;   // negative branch
        }
    }

       
    @Case("(1.0, 2.5) -> ok")
    @Case("(-1.0, 2.0) -> ok")
    @Case("(-1.0, -2.0) -> ok")
    public static void addDoubles(double a, double b) {
        double c = a + b;
        assert c > -1000000.0;
    }


    @Case("(\"foo\", \"bar\") -> ok")
    @Case("(\"\", \"baz\") -> ok")
    public static void concatStrings(String a, String b) {
        String s = a + b;
        assert s.length() >= b.length();
    }

  
    @Case("(-5, -5) -> ok")
    public static void addSigns(int a, int b) {
        int c = a + b;
        assert (c > -1000000);  
    }


    @Case("(5) -> ok")
    @Case("(0) -> assertion error")
    @Case("(-5) -> assertion error")
    public static void requirePositive(int x) {
        assert x > 0;
    }

}
