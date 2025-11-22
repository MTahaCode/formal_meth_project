from collections import defaultdict
from lark import Lark, Transformer, v_args, Tree
import re

class SSATransformer:
    def __init__(self):
        self.counter = defaultdict(int)
        self.env_stack = [{}]
        # track array slot versioning: array_name -> index_key -> version_counter
        self.array_versions = defaultdict(lambda: defaultdict(int))

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
        if not isinstance(expr, tuple):
            return expr
        op, *args = expr
        if op == 'num':
            return expr
        if op == 'var':
            return ('var', self.lookup(args[0]))
        if op == 'arr_access':
            base_t = self.transform_expr(args[0])
            idx_t  = self.transform_expr(args[1])
            # resolve slot var name: dynamic or constant index
            arr_base = base_t[1]
            if idx_t[0] == 'num':
                idx_key = str(idx_t[1])
            else:
                idx_key = idx_t[1]
            # lookup latest version
            ver = self.array_versions[arr_base].get(idx_key, 0)
            slot_base = f"{arr_base}_{idx_key}"
            if ver > 0:
                return ('var', f"{slot_base}_{ver}")
            else:
                # not yet written: fall back to initial slot var
                return ('var', slot_base)
        if op in ('arith','cmp'):
            head = self.transform_expr(args[0])
            rest = [(sym, self.transform_expr(rhs)) for sym, rhs in args[1]]
            return (op, head, rest)
        # other ops
        return tuple(self.transform_expr(a) for a in expr)

    def transform_stmt(self, stmt):
        op = stmt[0]
        # array literal explosion
        if op == 'assign' and stmt[2][0] == 'array_literal':
            base = stmt[1][1]
            items = stmt[2][1:]
            out = []
            for i, item in enumerate(items):
                key = f"{base}_{i}"
                self.env_stack[-1][key] = key
                out.append(('assign', ('var', key), item))
            return out

        if op == 'assign':
            lhs, rhs = stmt[1], stmt[2]
            rhs_t = self.transform_expr(rhs)
            # var write
            if lhs[0] == 'var':
                new = self.fresh(lhs[1])
                return ('assign', ('var', new), rhs_t)
            # arr write
            if lhs[0] == 'arr_access':
                base_t = self.transform_expr(lhs[1])
                idx_orig = lhs[2]
                idx_t = self.transform_expr(idx_orig)
                # determine index key
                if idx_t[0] == 'num':
                    idx_key = str(idx_t[1])
                else:
                    idx_key = idx_t[1]
                arr_base = base_t[1]
                # bump version
                ver = self.array_versions[arr_base]
                ver[idx_key] += 1
                slot_base = f"{arr_base}_{idx_key}"
                slot_name = f"{slot_base}_{ver[idx_key]}"
                # bind in env
                self.env_stack[-1][slot_base] = slot_name
                return ('assign', ('var', slot_name), rhs_t)

        # pass-through for other statements
        return stmt

    def transform_block(self, block):
        out = []
        for st in block:
            if isinstance(st, list):
                out.extend(self.transform_block(st))
            else:
                r = self.transform_stmt(st)
                if isinstance(r, list): out.extend(r)
                else: out.append(r)
        return out

class SSAPhiCodeGen:
    def __init__(self, unroll=1):
        self.lines = []
        self.buffered_asserts = []
        self.indent = ""
        self.version = {}
        self.full_map = {}
        self.unroll = unroll

    def fresh(self, name):
        v = self.version.get(name, 0) + 1
        self.version[name] = v
        full = f"{name}_{v}"
        self.full_map[name] = full
        return full

    def emit(self, text):
        self.lines.append(f"{self.indent}{text}")

    def gen(self, stmts):
        for s in stmts:
            if s[0] == "assert":
                self.buffered_asserts.append(s)
            else:
                self.gen_stmt(s)
        for stmt in self.buffered_asserts:
            cond_s = self.gen_expr(stmt[1])
            self.emit(f"assert({cond_s});")
        return "\n".join(self.lines)

    def gen_stmt(self, stmt):
        kind = stmt[0]

        # handle constant-index array write
        if kind == "assign" and stmt[1][0] == 'arr_access':
            _, (_, arr_base), idx_expr = stmt[1]
            idx_str = self.gen_expr(idx_expr)
            store_name = f"{arr_base}_store"
            rhs = self.gen_expr(stmt[2])
            self.emit(f"{store_name}({idx_str}, {rhs})")
            return

        # simple scalar assign
        if kind == "assign":
            var, expr = stmt[1][1], stmt[2]
            tgt = self.fresh(var)
            rhs = self.gen_expr(expr)
            self.emit(f"{tgt} = {rhs}")
            return

        if kind == "if":
            cond, then, els = stmt[1], stmt[2], stmt[3] or []
            self.emit(f"# if {self.gen_expr(cond)}")
            self.indent += "  "
            for s in then:
                self.gen_stmt(s)
            self.indent = self.indent[:-2]
            phi = self.make_phi(then, els)
            if phi:
                for line in phi.split("\n"):
                    self.emit(line)
            return

        if kind == "for":
            # unpack stmt tuple correctly: ('for', init, cond, upd, body)
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
            return

        # other
        if kind != "assert":
            self.emit(f"# unsupported: {stmt}")

    def make_phi(self, block_a, block_b):
        defs = {b[1][1] for b in block_a + block_b if b[0] == "assign"}
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
        last = None
        for b in block:
            if b[0] == "assign" and b[1][1] == var:
                last = self.version.get(var, 0)
        return f"{var}_{last or 0}"

    def gen_expr(self, expr):
        kind = expr[0]
        if kind == 'num':
            return str(expr[1])
        if kind == 'var':
            base = expr[1]
            ver = self.version.get(base)
            return f"{base}_{ver}" if ver else base
        if kind == 'arr_access':
            _, (_, arr), idx = expr
            idx_s = self.gen_expr(idx)
            return f"{arr}[{idx_s}]"
        if kind == 'arith':
            out = self.gen_expr(expr[1])
            for op, e in expr[2]:
                out = f"{out} {op} {self.gen_expr(e)}"
            return out
        if kind in ('cmp', 'term'):
            return self.gen_expr(expr[1])
        return '<expr?>'

    def current_version_map(self):
        return dict(self.full_map)

    def latest(self, base):
        return self.full_map.get(base, base)

    def static_array_final(self, array_base_prefix):
        result = {}
        pat = re.compile(rf'^({re.escape(array_base_prefix)})_(\d+)$')
        for base, full in self.full_map.items():
            m = pat.match(base)
            if m:
                result[int(m.group(2))] = full
        return result
