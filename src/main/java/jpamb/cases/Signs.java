package jpamb.cases;

import jpamb.utils.*;
import static jpamb.utils.Tag.TagType.*;

public class Signs {

    @Case("() -> {+}")
    @Tag({ INTEGER_OVERFLOW })
    public static int returnPositive() {
        return 1;
    }
    
}
