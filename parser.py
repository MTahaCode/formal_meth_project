from lark import Lark, Transformer, v_args
from colorama import Fore, Style
from pprint import pprint

mini_lang_grammar = r'''
    ?start: stmt+

    ?stmt: assign 
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
    ?expr_cmp: expr_arith (("<"|">"|"<="|">="|"=="|"!=") expr_arith)*
    ?expr_arith: expr_term (("+"|"-") expr_term)*
    ?expr_term: expr_factor (("*"|"/"|"%") expr_factor)*
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

    COMMENT: /\/\/[^\r\n]*/
    %ignore COMMENT
    %import common.CNAME
    %import common.WS
    %ignore WS
'''

# AST Transformer
@v_args(inline=True)
class ASTTransformer(Transformer):
    def number(self, token):
        return ("num", int(token))

    def identifier(self, name):
        return ("var", str(name))

    def assign(self, var, val):
        return ("assign", var, val)

    def if_stmt(self, cond, then_block, else_block=None):
        return ("if", cond, then_block, else_block)

    def while_stmt(self, cond, block):
        return ("while", cond, block)

    def for_stmt(self, init, cond, update, block):
        return ("for", init, cond, update, block)

    def assert_stmt(self, expr):
        return ("assert", expr)

    def print_stmt(self, expr):
        return ("print", expr)

    def array_access(self, name, index):
        return ("arr_access", name, index)

    def not_expr(self, expr):
        return ("not", expr)

    def expr_or(self, *args):
        return ("or", *args) if len(args) > 1 else args[0]

    def expr_and(self, *args):
        return ("and", *args) if len(args) > 1 else args[0]

    def expr_cmp(self, *args):
        return args if len(args) == 1 else ("cmp", *args)

    def expr_arith(self, *args):
        return args if len(args) == 1 else ("arith", *args)

    def expr_term(self, *args):
        return args if len(args) == 1 else ("term", *args)

    def stmt(self, *args):
        return list(args)

    def comment(self, _):
        return ("comment",)

# Build parser
parser = Lark(mini_lang_grammar, parser="lalr", transformer=ASTTransformer())

# Example usage
def parse_code(code):
    tree = parser.parse(code)
    return tree

# Example input code
if __name__ == '__main__':
    code = """
        n := 3;
        arr := [1, 2, 3];
        for (i := 0; i < n; i := i + 1;) {
            for (j := 0; j < n - i - 1; j := j + 1;) {
                //if (arr[j] > arr[j+1]) {
                    temp := arr[j+1];
                    arr[j] := arr[j+1];
                    arr[j+1] := temp;
                //}
            }
        }
    """
    ast = parse_code(code)

    # printing the parse tree
    pprint("Parsed from code")
    pprint(ast)


