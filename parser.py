from lark import Lark, Transformer, v_args, Tree
from collections import defaultdict
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

@v_args(inline=True)
class ASTTransformer(Transformer):
    def number(self, token): return ("num", int(token))
    def identifier(self, name): return ("var", str(name))
    def array_access(self, name, index): return ("arr_access", name, index)
    def array(self, first, *rest): return ("array_literal", first, *rest)
    def assign(self, var, val): return ("assign", var, val)
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
    def assert_stmt(self, expr): return ("assert", expr)
    def print_stmt(self, expr): return ("print", expr)
    def expr_or(self, *args): return ("or",) + args if len(args)>1 else args[0]
    def expr_and(self, *args): return ("and",) + args if len(args)>1 else args[0]
    def expr_cmp(self, *args): return ("cmp",) + args if len(args)>1 else args[0]
    def expr_arith(self, *args): return ("arith",) + args if len(args)>1 else args[0]
    def expr_term(self, *args): return ("term",) + args if len(args)>1 else args[0]
    def not_expr(self, expr): return ("not", expr)
    def stmt(self, *args): return list(args)

class SSATransformer:
    def __init__(self):
        self.counter = defaultdict(int)
        self.env_stack = [{}]

    def fresh(self, name):
        self.counter[name] += 1
        ver = f"{name}_{self.counter[name]}"
        self.env_stack[-1][name] = ver
        return ver

    def lookup(self, name):
        for env in reversed(self.env_stack):
            if name in env:
                return env[name]
        return name

    def transform_expr(self, expr):
        t = expr[0]
        if t == 'num':
            return expr
        if t == 'var':
            return ('var', self.lookup(expr[1]))
        if t == 'arr_access':
            # Combine array name and index SSA version into a new var
            base_name = expr[1][1]
            idx_expr = expr[2]
            # transform the index expression first
            idx_ssa = self.transform_expr(idx_expr)
            # idx_ssa should be ('var', idx_version) or ('num', const)
            idx_name = idx_ssa[1]
            new_name = f"{base_name}_{idx_name}"
            # lookup or fresh if not yet exist
            if new_name not in self.env_stack[-1]:
                self.fresh(new_name)
            return ('var', self.lookup(new_name))
        if t in ('not','print','assert'):
            return (t, self.transform_expr(expr[1]))
        if t in ('or','and','cmp','arith','term'):
            return (t,) + tuple(self.transform_expr(e) for e in expr[1:])
        return expr

    def transform_stmt(self, stmt):
        if stmt[0] == 'assign':
            lhs, rhs = stmt[1], stmt[2]
            # array literal destructuring unchanged
            if rhs[0] == 'array_literal':
                stmts = []
                for i, elt in enumerate(rhs[1:]):
                    name = f"a{i}"
                    ver = self.fresh(name)
                    stmts.append(('assign', ('var', ver), elt))
                return stmts
            rhs_t = self.transform_expr(rhs)
            if lhs[0] == 'var':
                name = lhs[1]
                ver = self.fresh(name)
                return ('assign', ('var', ver), rhs_t)
            # assign to array element: treat as write to combined var
            if lhs[0] == 'arr_access':
                base = lhs[1][1]
                idx_t = self.transform_expr(lhs[2])
                idx_name = idx_t[1]
                new_name = f"{base}_{idx_name}"
                ver = self.fresh(new_name)
                return ('assign', ('var', ver), rhs_t)
        if stmt[0] in ('print','assert'):
            return (stmt[0], self.transform_expr(stmt[1]))
        if stmt[0] == 'if':
            cond = self.transform_expr(stmt[1])
            self.env_stack.append(self.env_stack[-1].copy())
            then_b = self.transform_block(stmt[2])
            self.env_stack.pop()
            self.env_stack.append(self.env_stack[-1].copy())
            else_b = self.transform_block(stmt[3]) if stmt[3] else None
            self.env_stack.pop()
            return ('if', cond, then_b, else_b)
        if stmt[0] == 'while':
            self.env_stack.append(self.env_stack[-1].copy())
            cond = self.transform_expr(stmt[1])
            body = self.transform_block(stmt[2])
            self.env_stack.pop()
            return ('while', cond, body)
        if stmt[0] == 'for':
            self.env_stack.append(self.env_stack[-1].copy())
            init = self.transform_stmt(stmt[1])
            cond = self.transform_expr(stmt[2])
            up = self.transform_stmt(stmt[3])
            body = self.transform_block(stmt[4])
            self.env_stack.pop()
            return ('for', init, cond, up, body)
        return stmt

    def transform_block(self, block):
        if isinstance(block, Tree): block = block.children
        out = []
        for s in block:
            res = self.transform_stmt(s)
            if isinstance(res, list): out.extend(res)
            else: out.append(res)
        return out

if __name__ == '__main__':
    parser = Lark(mini_lang_grammar, parser='lalr', transformer=ASTTransformer())
    code = """
        n := 3;
        arr := [1, 2, 3];
        for (i := 0; i < n; i := i + 1;) {
            for (j := 0; j < n - i - 1; j := j + 1;) {
                if (arr[j] > arr[j+1]) {
                    temp := arr[j+1];
                    arr[j] := arr[j+1];
                    arr[j+1] := temp;
                }
            }
        }
    """
    ast = parser.parse(code)
    pprint("Original AST:")
    pprint(ast)

    ssa = SSATransformer()
    ssa_ast = ssa.transform_block(ast.children if isinstance(ast, Tree) else ast)
    pprint("\nSSA AST:")
    pprint(ssa_ast)
