import re

class SSAtoSMTLib:
    """
    Convert SSA-style code (as a single string) into SMT-LIB v2 format.
    - Lines starting with '#' are ignored.
    - a + b / a - b become (+ a b) / (- a b) everywhere.
    - phi(a,b) → (ite true a b).
    - Supports x = y, assert(x OP y), array reads and writes, and simple ifs.
    """
    # recognize SSA names, numeric literals, and array accesses like p1_arr_p1_j_1+1
    SSA_NAME   = r"\w+_\d+"
    NUM        = r"\d+"
    INDEX_EXPR = rf"(?:{SSA_NAME}|{NUM})"

    # assignment of scalar or array store
    ASSIGN_RE = re.compile(rf"^(?P<lhs>{SSA_NAME})\s*=\s*(?P<rhs>.+)$")
    STORE_RE  = re.compile(rf"^(?P<arr>\w+)_store\s*\(\s*(?P<idx>{INDEX_EXPR})\s*,\s*(?P<val>{INDEX_EXPR})\s*\)$")

    # assert(cond)
    ASSERT_RE = re.compile(
        rf"^assert\(\s*(?P<lhs>{SSA_NAME}|{NUM}|{INDEX_EXPR})\s*"
        r"(?P<op>>=|<=|!=|=|>|<)\s*"
        rf"(?P<rhs>.+?)"           # until closing paren
        r"\)"
    )

    # phi and arith
    PHI_RE      = re.compile(r"phi\(\s*(?P<a>\w+_\d+)\s*,\s*(?P<b>\w+_\d+)\s*\)")
    ARITH_ANY_RE = re.compile(r"\b(?P<x>\w+_\d+|\d+)\s*(?P<op>[+\-])\s*(?P<y>\w+_\d+|\d+)\b")

    def __init__(self, ssa_text: str):
        self.lines       = ssa_text.splitlines()
        self.assigns     = []     # (lhs, rhs_smt)
        self.stores      = []     # (arr, idx, val)
        self.cmp_asserts = []     # (lhs, op, rhs_smt)
        self.if_conditions = []   # capture nested if-guards
        self.arrays      = set()  # track array names
        self._parse()

    def _parse(self):
        guard_stack = []
        for raw in self.lines:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue

            # 1) if-statement
            if line.startswith('if '):
                m = re.match(rf"if\s*(?P<lhs>{self.SSA_NAME})\s*(?P<op>>|<|>=|<=)\s*(?P<rhs>{self.SSA_NAME}|\d+)", line)
                if m:
                    cond = (m.group('lhs'), m.group('op'), m.group('rhs'))
                    guard_stack.append(cond)
                continue

            # 2) end-if placeholder (dedent or next unindented)
            # For simplicity, assume each if only affects next statement

            # 3) assert(...)?
            ma = self.ASSERT_RE.match(line)
            if ma:
                lhs, op, rhs = ma.group('lhs', 'op', 'rhs')
                rhs_smt = self._rewrite_phi(rhs)
                rhs_smt = self._rewrite_arith(rhs_smt)
                # apply active guard if any
                if guard_stack:
                    g_lhs, g_op, g_rhs = guard_stack.pop()
                    cond_smt = f"({g_op} {g_lhs} {g_rhs})"
                    self.cmp_asserts.append((cond_smt, '=>', f"({op} {lhs} {rhs_smt})"))
                else:
                    self.cmp_asserts.append((lhs, op, rhs_smt))
                continue

            # 4) scalar assignment
            m = self.ASSIGN_RE.match(line)
            if m:
                lhs, rhs = m.group('lhs', 'rhs')
                rhs_smt = self._rewrite_phi(rhs)
                rhs_smt = self._rewrite_arith(rhs_smt)
                self.assigns.append((lhs, rhs_smt))
                continue

            # 5) array store: arr_store(idx, val)
            ms = self.STORE_RE.match(line)
            if ms:
                arr, idx, val = ms.group('arr', 'idx', 'val')
                self.arrays.add(arr)
                self.stores.append((arr, idx, val))

    def _rewrite_phi(self, expr: str) -> str:
        return self.PHI_RE.sub(r"(ite true \g<a> \g<b>)", expr)

    def _rewrite_arith(self, expr: str) -> str:
        prev = None
        while prev != expr:
            prev = expr
            expr = self.ARITH_ANY_RE.sub(r"(\2 \g<x> \g<y>)", expr)
        return expr

    def to_smt2(self) -> str:
        # declare scalar and array vars
        decls = []
        # scalar SSA
        all_vars = set(v for v,_ in self.assigns) | {v for v,_ in self.cmp_asserts}
        for v in sorted(all_vars):
            decls.append(f"(declare-fun {v} () Int)")
        # arrays
        for arr in sorted(self.arrays):
            decls.append(f"(declare-fun {arr} () (Array Int Int))")
        parts = decls + ['']

        # asserts for assignments
        for lhs, rhs in self.assigns:
            parts.append(f"(assert (= {lhs} {rhs}))")

        # asserts for stores
        for arr, idx, val in self.stores:
            parts.append(f"(assert (= {arr} (store {arr} {idx} {val})))")

        # comparison asserts
        for lhs, op, rhs in self.cmp_asserts:
            if op == '=>':
                parts.append(f"(assert (=> {lhs} {rhs}))")
            else:
                parts.append(f"(assert ({op} {lhs} {rhs}))")

        parts += ['', '(check-sat)', '(get-model)']
        return "\n".join(parts)
