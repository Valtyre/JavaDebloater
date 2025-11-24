package jpamb.cases;

import jpamb.utils.Case;

public class Signs {

    // -------------------------
    // Basic sign-generating methods
    // -------------------------

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

    // -------------------------
    // Sign-based branching
    // -------------------------

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

    // -------------------------
    // Sign propagation through addition
    // -------------------------

    @Case("(5, 5) -> ok")
    @Case("(5, -5) -> ok")
    @Case("(-5, -5) -> ok")
    public static void addSigns(int a, int b) {
        int c = a + b;
        // Just asserts that addition terminates and is well-defined.
        assert (c > -1000000);  
    }

    // -------------------------
    // Assertion on required sign
    // -------------------------

    @Case("(5) -> ok")
    @Case("(0) -> assertion error")
    @Case("(-5) -> assertion error")
    public static void requirePositive(int x) {
        assert x > 0;
    }

}
