from collections import defaultdict
from lark import Lark, Transformer, v_args, Tree

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
        if isinstance(expr, list):
            return [self.transform_expr(e) for e in expr]
        if isinstance(expr, tuple):
            op, *args = expr
            if op == 'num':
                return expr
            if op == 'var':
                return ('var', self.lookup(args[0]))
            if op == 'arr_access':
                base_t = self.transform_expr(args[0])
                idx_t  = self.transform_expr(args[1])
                return ('arr_access', base_t, idx_t)
            if op in ('not', 'print', 'assert'):
                return (op, self.transform_expr(args[0]))
            if op in ('or', 'and'):
                return (op,) + tuple(self.transform_expr(a) for a in args)
            if op in ('cmp', 'arith', 'term'):
                head = self.transform_expr(args[0])
                rest = [(symb, self.transform_expr(rhs)) for symb, rhs in args[1]]
                return (op, head, rest)
            return tuple(self.transform_expr(e) for e in expr)
        return expr

    def transform_stmt(self, stmt):
        op = stmt[0]

        # expand array literals into individual element assignments
        if op == 'assign' and stmt[2][0] == 'array_literal':
            base = stmt[1][1]
            items = stmt[2][1:]
            out = []
            for i, item in enumerate(items):
                key = f"{base}_{i}"
                self.env_stack[-1][key] = key
                out.append(('assign', ('var', key), self.transform_expr(item)))
            return out

        # standard assignment
        if op == 'assign':
            lhs, rhs = stmt[1], stmt[2]
            rhs_t = self.transform_expr(rhs)

            # simple variable assignment
            if lhs[0] == 'var':
                new = self.fresh(lhs[1])
                return ('assign', ('var', new), rhs_t)

            # array write turns into SSA'd pseudo-variables
            if lhs[0] == 'arr_access':
                base_t   = self.transform_expr(lhs[1])
                orig_idx = lhs[2]
                idx_t    = self.transform_expr(orig_idx)
                stmts    = []

                # derive a clean index name
                if orig_idx[0] == 'var':
                    orig_name = orig_idx[1]
                elif orig_idx[0] == 'arith' and isinstance(orig_idx[1], tuple) and orig_idx[1][0] == 'var':
                    orig_name = orig_idx[1][1]
                else:
                    orig_name = 'idx'

                # if index expression is complex, bind it
                if idx_t[0] != 'var' or idx_t[1] != orig_name:
                    idx_var = self.fresh(orig_name)
                    stmts.append(('assign', ('var', idx_var), idx_t))
                    idx_t = ('var', idx_var)

                # use underscore instead of brackets for SSA name
                slot_key = f"{base_t[1]}_{idx_t[1]}"
                new_slot = self.fresh(slot_key)
                stmts.append(('assign', ('var', new_slot), rhs_t))
                return stmts

        # prints and asserts
        if op in ('print', 'assert'):
            return (op, self.transform_expr(stmt[1]))

        # if statements
        if op == 'if':
            cond = self.transform_expr(stmt[1])
            self.env_stack.append(self.env_stack[-1].copy())
            then_b = self.transform_block(stmt[2])
            self.env_stack.pop()

            else_b = None
            if stmt[3]:
                self.env_stack.append(self.env_stack[-1].copy())
                else_b = self.transform_block(stmt[3])
                self.env_stack.pop()

            return ('if', cond, then_b, else_b)

        # while loops
        if op == 'while':
            self.env_stack.append(self.env_stack[-1].copy())
            cond = self.transform_expr(stmt[1])
            body = self.transform_block(stmt[2])
            self.env_stack.pop()
            return ('while', cond, body)

        # for loops
        if op == 'for':
            self.env_stack.append(self.env_stack[-1].copy())
            init = self.transform_stmt(stmt[1])
            cond = self.transform_expr(stmt[2])
            upd  = self.transform_stmt(stmt[3])
            body = self.transform_block(stmt[4])
            self.env_stack.pop()
            return ('for', init, cond, upd, body)

        # fallback: return as-is
        return stmt

    def transform_block(self, block):
        out = []
        for st in block:
            if isinstance(st, list):
                out.extend(self.transform_block(st))
            else:
                r = self.transform_stmt(st)
                if isinstance(r, list):
                    out.extend(r)
                else:
                    out.append(r)
        return out


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
            self.emit(f"# if {self.gen_expr(cond)}")
            self.indent += "  "
            for s in then:
                self.gen_stmt(s)
            self.indent = self.indent[:-2]

            if els:
                self.emit("# else")
                self.indent += "  "
                for s in els:
                    self.gen_stmt(s)
                self.indent = self.indent[:-2]

            phi = self.make_phi(then, els)
            if phi:
                for line in phi.split("\n"):
                    self.emit(line)

        elif kind == "while":
            cond = stmt[1]
            body = stmt[2]
            self.emit(f"# while {self.gen_expr(cond)}")
            for k in range(self.unroll):
                self.indent += "  "
                for s in body:
                    self.gen_stmt(s)
                self.indent = self.indent[:-2]
                self.emit(f"# unroll {k+1}")
            phi = self.make_phi(body, [])
            if phi:
                for line in phi.split("\n"):
                    self.emit(line)

        elif kind == "for":
            init, cond, upd, body = stmt[1], stmt[2], stmt[3], stmt[4]
            self.gen_stmt(init)
            self.emit(f"# for-loop while {self.gen_expr(cond)}")
            for k in range(self.unroll):
                self.indent += "  "
                for s in body:
                    self.gen_stmt(s)
                self.gen_stmt(upd)
                self.indent = self.indent[:-2]
                self.emit(f"# unroll {k+1}")
            phi = self.make_phi(body, [init])
            if phi:
                for line in phi.split("\n"):
                    self.emit(line)

        else:
            self.emit(f"# unsupported: {stmt}")

    def make_phi(self, block_a, block_b):
        defs = set()
        for b in block_a + block_b:
            if (isinstance(b, tuple) and b[0] == "assign"
                    and isinstance(b[1], tuple) and isinstance(b[1][1], str)):
                defs.add(b[1][1])

        if not defs:
            return ""

        lines = []
        for var in sorted(defs):
            v1 = self.last_def(var, block_a)
            v2 = self.last_def(var, block_b)
            res = self.fresh(var)
            lines.append(f"{res} = phi({v1}, {v2})")
        return "\n".join(lines)

    def last_def(self, var, block):
        last_ver = None
        for b in block:
            if (isinstance(b, tuple)
                    and b[0] == "assign"
                    and isinstance(b[1], tuple)
                    and b[1][1] == var):
                last_ver = self.version.get(var, last_ver)
        return f"{var}_{last_ver or 0}"

    def gen_expr(self, expr):
        kind = expr[0]
        if kind == "num":
            return str(expr[1])
        if kind == "var":
            return expr[1]

        # **UPDATED**: render arr_access as `base_idx`
        if kind == "arr_access":
            base = self.gen_expr(expr[1])
            idx  = self.gen_expr(expr[2])
            return f"{base}_{idx}"

        if kind in ("arith", "cmp", "term"):
            out = self.gen_expr(expr[1])
            for op, e in expr[2]:
                out = f"{out} {op} {self.gen_expr(e)}"
            return out

        if kind == "not":
            return f"!{self.gen_expr(expr[1])}"
        if kind in ("and", "or"):
            op = "&&" if kind == "and" else "||"
            return op.join(self.gen_expr(e) for e in expr[1:])

        return "<expr?>"
