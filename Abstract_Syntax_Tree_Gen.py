from lark import Lark, Transformer, v_args, Tree

mini_lang_grammar = r'''
    ?start: stmt+

    ?stmt: assign                -> stmt
         | if_stmt
         | while_stmt
         | for_stmt
         | assert_stmt
         | print_stmt

    stmt_block: stmt+

    assign: (identifier | array_access) ":=" expr ";"

    if_stmt: "if" "(" expr ")" "{" stmt_block "}" ("else" "{" stmt_block "}")?
    while_stmt: "while" "(" expr ")" "{" stmt_block "}"
    for_stmt: "for" "(" assign expr ";" assign ")" "{" stmt_block "}"
    assert_stmt: "assert" "(" expr ")" ";"
    print_stmt: "print" "(" expr ")" ";"

    ?expr: expr_or

    ?expr_or: expr_and ("||" expr_and)*
    ?expr_and: expr_cmp ("&&" expr_cmp)*

    ?expr_cmp: expr_arith (COMP_OP expr_arith)*
    ?expr_arith: expr_term (ADD_OP expr_term)*
    ?expr_term: expr_factor (MUL_OP expr_factor)*

    ?expr_factor: NUMBER               -> number
                | identifier
                | array_access
                | array_literal
                | "(" expr ")"
                | "!" expr             -> not_expr

    array_access: identifier "[" expr "]"
    array_literal: "[" expr ("," expr)* "]"   -> array

    identifier: CNAME
    NUMBER: /[0-9]+/

    // operator tokens
    COMP_OP: "<=" | ">=" | "<" | ">" | "==" | "!="
    ADD_OP: "+" | "-"
    MUL_OP: "*" | "/" | "%"

    COMMENT: /\/\/[^\r\n]*/
    %ignore COMMENT
    %import common.CNAME
    %import common.WS
    %ignore WS
'''
@v_args(inline=True)
class ASTTransformer(Transformer):
    def number(self, token):
        return ("num", int(token))

    def identifier(self, name):
        return ("var", str(name))

    def array_access(self, name, index):
        return ("arr_access", name, index)

    def array(self, first, *rest):
        return ("array_literal", first, *rest)

    def assign(self, var, val):
        return ("assign", var, val)

    def if_stmt(self, cond, then_block, else_block=None):
        then = then_block.children if isinstance(then_block, Tree) else then_block
        els = else_block.children if isinstance(else_block, Tree) else else_block
        return ("if", cond, then, els)

    def while_stmt(self, cond, block):
        blk = block.children if isinstance(block, Tree) else block
        return ("while", cond, blk)

    def for_stmt(self, init, cond, update, block):
        blk = block.children if isinstance(block, Tree) else block
        return ("for", init, cond, update, blk)

    def assert_stmt(self, expr):
        return ("assert", expr)

    def print_stmt(self, expr):
        return ("print", expr)

    def expr_or(self, first, *rest):
        return ("or", first, *rest) if rest else first

    def expr_and(self, first, *rest):
        return ("and", first, *rest) if rest else first

    def expr_cmp(self, first, *rest):
        # rest: op1, val1, op2, val2, ...
        pairs = [(rest[i].value, rest[i+1]) for i in range(0, len(rest), 2)]
        return ("cmp", first, pairs)

    def expr_arith(self, first, *rest):
        pairs = [(rest[i].value, rest[i+1]) for i in range(0, len(rest), 2)]
        return ("arith", first, pairs)

    def expr_term(self, first, *rest):
        pairs = [(rest[i].value, rest[i+1]) for i in range(0, len(rest), 2)]
        return ("term", first, pairs)

    def not_expr(self, expr):
        return ("not", expr)

    def stmt(self, *args):
        return list(args)

abstract_syntax_tree_parser=Lark(mini_lang_grammar,parser='lalr',transformer=ASTTransformer())

