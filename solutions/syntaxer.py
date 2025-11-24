#!/usr/bin/env python3
"""A very stupid syntatic analysis, that only checks for assertion errors."""

import logging
import tree_sitter
import tree_sitter_java
import jpamb
import sys
from pathlib import Path


methodid = jpamb.getmethodid(
    "syntaxer",
    "1.0",
    "The Rice Theorem Cookers",
    ["syntatic", "python"],
    for_science=True,
)


JAVA_LANGUAGE = tree_sitter.Language(tree_sitter_java.language())
parser = tree_sitter.Parser(JAVA_LANGUAGE)

log = logging
log.basicConfig(level=logging.DEBUG)


srcfile = jpamb.sourcefile(methodid).relative_to(Path.cwd())

with open(srcfile, "rb") as f:
    log.debug("parse sourcefile %s", srcfile)
    tree = parser.parse(f.read())

simple_classname = str(methodid.classname.name)

log.debug(f"{simple_classname}")

# To figure out how to write these you can consult the
# https://tree-sitter.github.io/tree-sitter/playground
class_q = tree_sitter.Query(
    JAVA_LANGUAGE,
    f"""
    (class_declaration 
        name: ((identifier) @class-name 
               (#eq? @class-name "{simple_classname}"))) @class
""",
)

for node in tree_sitter.QueryCursor(class_q).captures(tree.root_node)["class"]:
    break
else:
    log.error(f"could not find a class of name {simple_classname} in {srcfile}")

    sys.exit(-1)

# log.debug("Found class %s", node.range)

method_name = methodid.extension.name

method_q = tree_sitter.Query(
    JAVA_LANGUAGE,
    f"""
    (method_declaration name: 
      ((identifier) @method-name (#eq? @method-name "{method_name}"))
    ) @method
""",
)

for node in tree_sitter.QueryCursor(method_q).captures(node)["method"]:

    if not (p := node.child_by_field_name("parameters")):
        log.debug(f"Could not find parameteres of {method_name}")
        continue

    params = [c for c in p.children if c.type == "formal_parameter"]

    if len(params) != len(methodid.extension.params):
        continue

    log.debug(methodid.extension.params)
    log.debug(params)

    for tn, t in zip(methodid.extension.params, params):
        if (tp := t.child_by_field_name("type")) is None:
            break

        if tp.text is None:
            break

        # todo check for type.
    else:
        break
else:
    log.warning(f"could not find a method of name {method_name} in {simple_classname}")
    sys.exit(-1)

log.debug("Found method %s %s", method_name, node.range)

body = node.child_by_field_name("body")
assert body and body.text
for t in body.text.splitlines():
    log.debug("line: %s", t.decode())

assert_q = tree_sitter.Query(JAVA_LANGUAGE, """(assert_statement) @assert""")
assert_false_q = tree_sitter.Query(
    JAVA_LANGUAGE, 
    """
      ((assert_statement (false) @assert-false))
      """)
assert_true_q = tree_sitter.Query(
    JAVA_LANGUAGE, 
    """
      ((assert_statement (true) @assert-true))
      """)

divide_by_zero_q = tree_sitter.Query(
    JAVA_LANGUAGE, 
    """(binary_expression operator: "/" right: (decimal_integer_literal) @rhs(#eq? @rhs "0")) 
    @divide_by_zero""")


#todo handle a / b cases or 1 / n cases
divide_by_one_variable_q = tree_sitter.Query(
    JAVA_LANGUAGE, 
    """(binary_expression operator: "/" right: (identifier) @rhs) 
    @divide_by_one_variable""")


infinite_loop_q = tree_sitter.Query(
    JAVA_LANGUAGE,
    """
    (
      (while_statement
        condition: (parenthesized_expression
                     (true))
        body: (_) @loop-body
      ) @infinite-loop
    )
    """)

null_array_decl_q = tree_sitter.Query(
    JAVA_LANGUAGE,
    """
    (
      (local_variable_declaration
        declarator: (variable_declarator
          value: (null_literal) @null-val
        )
      ) @null-array-decl
    )
    """)

array_access_q = tree_sitter.Query(
    JAVA_LANGUAGE,
    """
    (
      (array_access
        array: (identifier) @array-name
        index: (_) @field_value
      ) @array-access
    )
    """
)

#todo: handle new int[]
array_access_with_new_q = tree_sitter.Query(
    JAVA_LANGUAGE,
    """
    (
      (array_access
        array: (new_array
                 type: (identifier) @array-type
                 size: (_) @array-size
               ) 
        index: (_) @field_value
      ) @array-access-new
    )
    """
)

array_length_q = tree_sitter.Query(
    JAVA_LANGUAGE,
    """
    (
      (field_access
        object: (identifier) @array-name
        field: (identifier) @length-field (#eq? @length-field "length")
      ) @array-length
    )
    """
)




#return_nothing_q = tree_sitter.Query(
#    JAVA_LANGUAGE,
#    """((method_declaration
#        type: (void_type)
#        body: (block
#          (return_statement) @void-return)) @method)"""
#)



null_array_decl_found = any(
    capture_name == "null-array-decl"
    for capture_name, _ in tree_sitter.QueryCursor(null_array_decl_q).captures(body).items()
)

array_access_found = any(
    capture_name == "array-access"
    for capture_name, _ in tree_sitter.QueryCursor(array_access_q).captures(body).items()
)

array_length_found = any(
    capture_name == "array-length"
    for capture_name, _ in tree_sitter.QueryCursor(array_length_q).captures(body).items()
)

assert_found = any(
    capture_name == "assert"
    for capture_name, _ in tree_sitter.QueryCursor(assert_q).captures(body).items()
)

assert_false_found = any(
    capture_name == "assert-false"
    for capture_name, _ in tree_sitter.QueryCursor(assert_false_q).captures(body).items()
)

assert_true_found = any(
    capture_name == "assert-true"
    for capture_name, _ in tree_sitter.QueryCursor(assert_true_q).captures(body).items()
)

divide_by_zero_found = any(
    capture_name == "divide_by_zero"
    for capture_name, _ in tree_sitter.QueryCursor(divide_by_zero_q).captures(body).items()
)

divide_by_one_variable_found = any(
    capture_name == "divide_by_one_variable"
    for capture_name, _ in tree_sitter.QueryCursor(divide_by_one_variable_q).captures(body).items()
)

infinite_loop_found = any(
    capture_name == "infinite-loop"
    for capture_name, _ in tree_sitter.QueryCursor(infinite_loop_q).captures(body).items()
)



if null_array_decl_found and (array_access_found or array_length_found):
    log.debug("Found null array dereference")
    print("null pointer;70%")
else:
    log.debug("No null array dereference")
    print("null pointer;0%")

if infinite_loop_found:
    log.debug("Found infinite loop")
    print("*;100%")
else:
    log.debug("No infinite loop")
    print("*;0%")

if divide_by_zero_found:
    log.debug("Found divide by zero")
    print("divide by zero error;100%")
else:
    log.debug("No divide by zero")
    print("divide by zero error;0%")

if assert_false_found:
    log.debug("Found false assertion")
    print("assertion error;100%")
elif assert_found:
    log.debug("Found assertion")
    print("assertion error;50%")
else:
    log.debug("No assertion")
    print("assertion error;0%")

sys.exit(0)