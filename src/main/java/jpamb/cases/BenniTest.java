package jpamb.cases;

import jpamb.utils.Case;

public class BenniTest {

    @Case("() -> ok")
    public static void testSomething() {
        assert true;
    }

    @Case("(5) -> ok")
    @Case("(0) -> assertion error")
    public static void testInput(int x) {
        assert x > 0;
    }
}
