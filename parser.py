from lark import Lark, Transformer, v_args, Tree
from collections import defaultdict
from pprint import pprint
import re

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

# Build the parser
tparser = Lark(mini_lang_grammar, parser='lalr', propagate_positions=True)

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
        if t == 'num': return expr
        if t == 'var': return ('var', self.lookup(expr[1]))
        if t == 'arr_access':
            base = expr[1][1]
            idx = self.transform_expr(expr[2])
            idx_name = idx[1]
            new = f"{base}_{idx_name}"
            if new not in self.env_stack[-1]: self.fresh(new)
            return ('var', self.lookup(new))
        if t in ('not','print','assert'): return (t, self.transform_expr(expr[1]))
        if t in ('or','and','cmp','arith','term'):
            return (t,) + tuple(self.transform_expr(e) for e in expr[1:])
        return expr

    def transform_stmt(self, stmt):
        if stmt[0] == 'assign':
            lhs, rhs = stmt[1], stmt[2]
            if rhs[0] == 'array_literal':
                stmts=[]
                for i,elt in enumerate(rhs[1:]):
                    v = self.fresh(f"a{i}")
                    stmts.append(('assign',('var',v),elt))
                return stmts
            rhs_t = self.transform_expr(rhs)
            if lhs[0]=='var': return ('assign',('var',self.fresh(lhs[1])),rhs_t)
            # write to array element
            base, idx = lhs[1][1], self.transform_expr(lhs[2])
            name = f"{base}_{idx[1]}"
            return ('assign',('var',self.fresh(name)),rhs_t)
        if stmt[0] in ('print','assert'): return (stmt[0], self.transform_expr(stmt[1]))
        if stmt[0]=='if':
            cond=self.transform_expr(stmt[1])
            self.env_stack.append(self.env_stack[-1].copy())
            then=self.transform_block(stmt[2])
            self.env_stack.pop()
            self.env_stack.append(self.env_stack[-1].copy())
            els=self.transform_block(stmt[3]) if stmt[3] else None
            self.env_stack.pop()
            return ('if',cond,then,els)
        if stmt[0]=='while':
            self.env_stack.append(self.env_stack[-1].copy())
            cond=self.transform_expr(stmt[1])
            body=self.transform_block(stmt[2])
            self.env_stack.pop()
            return ('while',cond,body)
        if stmt[0]=='for':
            self.env_stack.append(self.env_stack[-1].copy())
            init=self.transform_stmt(stmt[1])
            cond=self.transform_expr(stmt[2])
            up=self.transform_stmt(stmt[3])
            body=self.transform_block(stmt[4])
            self.env_stack.pop()
            return ('for',init,cond,up,body)
        return stmt

    def transform_block(self,block):
        if isinstance(block,Tree): block=block.children
        out=[]
        for s in block:
            r=self.transform_stmt(s)
            out.extend(r if isinstance(r,list) else [r])
        return out

class SSACodeGen:
    def __init__(self):
        self.lines = []
        self.indent_level = 0

    def emit(self, line):
        indent = "  " * self.indent_level
        self.lines.append(f"{indent}{line}")

    def gen_expr(self, expr):
        kind = expr[0]
        if kind == "num":
            return str(expr[1])
        if kind == "var":
            return expr[1]
        if kind == "not":
            return f"!{self.gen_expr(expr[1])}"
        if kind in ("or", "and"):
            op = "||" if kind == "or" else "&&"
            return f" {op} ".join(self.gen_expr(e) for e in expr[1:])
        if kind == "cmp":
            left = self.gen_expr(expr[1])
            for op, right in expr[2]:
                left = f"{left} {op} {self.gen_expr(right)}"
            return left
        if kind in ("arith", "term"):
            left = self.gen_expr(expr[1])
            for op, right in expr[2]:
                left = f"{left} {op} {self.gen_expr(right)}"
            return left
        if kind == "arr_access":
            return f"{expr[1][1]}_{self.gen_expr(expr[2])}"
        return "<unsupported_expr>"

    def gen_stmt(self, stmt):
        kind = stmt[0]
        if kind == "assign":
            var = stmt[1][1]
            expr = self.gen_expr(stmt[2])
            self.emit(f"{var} = {expr}")
        elif kind == "print":
            self.emit(f"print {self.gen_expr(stmt[1])}")
        elif kind == "assert":
            self.emit(f"assert {self.gen_expr(stmt[1])}")
        elif kind == "if":
            cond = self.gen_expr(stmt[1])
            self.emit(f"if {cond}:")
            self.indent_level += 1
            for s in stmt[2]:
                self.gen_stmt(s)
            self.indent_level -= 1
            if stmt[3]:
                self.emit("else:")
                self.indent_level += 1
                for s in stmt[3]:
                    self.gen_stmt(s)
                self.indent_level -= 1
        elif kind == "while":
            cond = self.gen_expr(stmt[1])
            self.emit(f"while {cond}:")
            self.indent_level += 1
            for s in stmt[2]:
                self.gen_stmt(s)
            self.indent_level -= 1
            self.emit("# unrolled 1")
        elif kind == "for":
            self.gen_stmt(stmt[1])  # init
            cond = self.gen_expr(stmt[2])
            self.emit(f"while {cond}:")
            self.indent_level += 1
            for s in stmt[4]:
                self.gen_stmt(s)
            self.gen_stmt(stmt[3])  # update
            self.indent_level -= 1
            self.emit("# unrolled 1")
        else:
            self.emit(f"# unsupported stmt: {stmt}")

    def generate(self, stmts):
        for s in stmts:
            self.gen_stmt(s)
        return "\n".join(self.lines)

class SSAPhiCodeGen:
    def __init__(self, unroll=1):
        self.lines = []
        self.indent = ""
        self.version = {}
        self.unroll = unroll

    def fresh(self, name):
        v = self.version.get(name, 0) + 1
        self.version[name] = v
        return f"{name}_{v}"

    def emit(self, text):
        self.lines.append(f"{self.indent}{text}")

    def gen(self, stmts):
        for s in stmts:
            self.gen_stmt(s)
        return "\n".join(self.lines)

    def gen_stmt(self, stmt):
        kind = stmt[0]
        if kind == "assign":
            var, expr = stmt[1][1], stmt[2]
            tgt = self.fresh(var)
            rhs = self.gen_expr(expr)
            self.emit(f"{tgt} = {rhs}")
        elif kind == "if":
            cond, then, els = stmt[1], stmt[2], stmt[3] or []
            # unroll: first the taken branch
            self.emit(f"# if {self.gen_expr(cond)}")
            self.indent += "  "
            for s in then:
                self.gen_stmt(s)
            self.indent = self.indent[:-2]
            # then the else branch
            if els:
                self.emit("# else")
                self.indent += "  "
                for s in els:
                    self.gen_stmt(s)
                self.indent = self.indent[:-2]
            # φ-node for any var assigned in both
            self.emit(self.make_phi(then, els))
        elif kind == "while":
            init, cond, body = stmt[1], stmt[2], stmt[4]
            # emit the precondition check
            self.gen_stmt(init)
            header = self.gen_expr(cond)
            self.emit(f"# while {header}")

            # now unroll 'body' + phi-update 'unroll' times
            for k in range(self.unroll):
                self.indent += "  "
                for s in body:
                    self.gen_stmt(s)
                self.indent = self.indent[:-2]
                self.emit(f"# unroll {k+1}")

            # finally merge back with a phi
            self.emit(self.make_phi(body, [init]))

        elif kind == "for":
            init, cond, update, body = stmt[1], stmt[2], stmt[3], stmt[4]
            self.gen_stmt(init)
            header = self.gen_expr(cond)
            self.emit(f"# for-loop while {header}")

            for k in range(self.unroll):
                self.indent += "  "
                for s in body:
                    self.gen_stmt(s)
                # at end of each unroll, do the loop update
                self.gen_stmt(update)
                self.indent = self.indent[:-2]
                self.emit(f"# unroll {k+1}")

            self.emit(self.make_phi(body, [init]))
        else:
            self.emit(f"# unsupported: {stmt}")

    def make_phi(self, block_a, block_b):
        # collect all *string* var-names assigned in either block
        defs = set()
        for b in (block_a + block_b):
            if (isinstance(b, tuple)
            and b[0] == "assign"
            and isinstance(b[1], tuple)
            and isinstance(b[1][1], str)):
                defs.add(b[1][1])

        if not defs:
            return ""

        lines = []
        for var in sorted(defs):  # now all are strings
            v1 = self.last_def(var, block_a)
            v2 = self.last_def(var, block_b)
            res = self.fresh(var)
            lines.append(f"{res} = phi({v1}, {v2})")
        return "\n".join(lines)

    def last_def(self, var, block):
        # Find the *last* version of `var` in `block`
        last_ver = None
        for b in block:
            if (isinstance(b, tuple)
            and b[0] == "assign"
            and isinstance(b[1], tuple)
            and b[1][1] == var):
                # peek at what version `fresh()` gave it
                # we know self.version[var] was incremented on that fresh()
                last_ver = self.version.get(var, last_ver)
        return f"{var}_{last_ver or 0}"

    def gen_expr(self, expr):
        kind = expr[0]
        if kind=="num": return str(expr[1])
        if kind=="var": return expr[1]
        if kind in ("arith","cmp","term"):
            out = self.gen_expr(expr[1])
            for op, e in expr[2]:
                out = f"{out} {op} {self.gen_expr(e)}"
            return out
        if kind=="arr_access":
            base = expr[1][1]
            idx = self.gen_expr(expr[2])
            return f"{base}_{idx}"
        if kind=="not":
            return f"!{self.gen_expr(expr[1])}"
        if kind in ("and","or"):
            op = "&&" if kind=="and" else "||"
            return f" {op} ".join(self.gen_expr(e) for e in expr[1:])
        return "<expr?>"

class SSA2SMT:
    """
    Generate an SMT-LIB script from an SSA-transformed AST,
    with loops unrolled a fixed number of times.
    """
    def __init__(self, unroll_bound=1):
        self.unroll_bound = unroll_bound

    def expr_to_smt(self, e):
        """Recursively emit an SMT-LIB prefix expression from an SSA AST node."""
        kind = e[0]
        if kind == "num":
            return str(e[1])
        if kind == "var":
            return e[1]
        if kind in ("arith", "term"):
            out = self.expr_to_smt(e[1])
            for op, r in e[2]:
                smt_op = {"+":"+","-":"-","*":"*","/":"/","%":"mod"}[op]
                out = f"({smt_op} {out} {self.expr_to_smt(r)})"
            return out
        if kind == "cmp":
            out = self.expr_to_smt(e[1])
            for op, r in e[2]:
                smt_op = {"<":"<",">":">","<=":"<=",">=":">=","==":"=","!=":"distinct"}[op]
                out = f"({smt_op} {out} {self.expr_to_smt(r)})"
            return out
        if kind == "not":
            return f"(not {self.expr_to_smt(e[1])})"
        if kind == "and":
            return f"(and {' '.join(self.expr_to_smt(x) for x in e[1:])})"
        if kind == "or":
            return f"(or {' '.join(self.expr_to_smt(x) for x in e[1:])})"
        if kind == "phi":  # optional if phi-nodes are present
            # e = ("phi", val_true, val_false, cond)
            return f"(ite {self.expr_to_smt(e[3])} {self.expr_to_smt(e[1])} {self.expr_to_smt(e[2])})"
        raise ValueError(f"Unknown expr kind: {kind}")

    def generate(self, ssa_ast):
        """
        Unroll loops in SSA AST, collect all SSA variables,
        and emit a complete SMT-LIB QF_LIA script as a string.
        """
        # 1) Linearize SSA with unrolling
        phi_gen = SSAPhiCodeGen(unroll=self.unroll_bound)
        linear = phi_gen.gen(ssa_ast).splitlines()

        # 2) Collect all SSA names foo_N
        symbol_re = re.compile(r"\b([A-Za-z_]\w*_[0-9]+)\b")
        syms = set(symbol_re.findall("\n".join(linear)))

        # 3) Start SMT-LIB
        smt = ["(set-logic QF_LIA)"]
        # declare phi if needed
        smt.append("(declare-fun phi (Int Int Bool) Int)")
        # declare all vars
        for v in sorted(syms):
            smt.append(f"(declare-fun {v} () Int)")

        # 4) Emit asserts
        for raw in linear:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # assignment
            m = re.match(r"^(\w+_[0-9]+)\s*=\s*(.+)$", line)
            if m:
                var, rhs_str = m.groups()
                # we assume `rhs_str` is already in prefix form via expr_to_smt
                smt.append(f"(assert (= {var} {rhs_str}))")
            # original asserts
            elif line.startswith("assert "):
                cond = line[len("assert "):].strip()
                smt.append(f"(assert {cond})")

        smt.append("(check-sat)")
        return "\n".join(smt)

if __name__=='__main__':
    parser=Lark(mini_lang_grammar,parser='lalr',transformer=ASTTransformer())
    code="""
        n := 3;
    """
    ast=parser.parse(code)
    # print(ast)

    ssa=SSATransformer()
    ssa_ast=ssa.transform_block(ast.children if isinstance(ast,Tree) else ast)
    # print(ssa_ast)

    ssa_gen = SSA2SMT(unroll_bound=3)
    smt_script = ssa_gen.generate(ssa_ast)
    print(smt_script)

    # ssa_intermediate_gen=SSACodeGen()
    # ssa_internmediate_code = ssa_intermediate_gen.generate(ssa_ast)
    # # print(gen.generate(ssa_ast))

    # ssa_gen = SSAPhiCodeGen(unroll=3)
    # print(ssa_gen.gen(ssa_ast))

