import re

class SSAtoSMTLib:
    """
    Convert SSA-style code (as a single string) into SMT-LIB v2 format.
    - Lines starting with '#' are ignored.
    - `a + b` / `a - b` become `(+ a b)` / `(- a b)` everywhere.
    - `phi(a,b)` → `(ite true a b)`.
    - Supports both `x = y` and `assert(x OP y);` lines, even if y is `x_1 + 1`.
    """
    ASSIGN_RE = re.compile(r"^(?P<lhs>\w+_\d+)\s*=\s*(?P<rhs>.+)$")
    ASSERT_RE = re.compile(
        r"^assert\(\s*"
        r"(?P<lhs>\w+_\d+|\d+)\s*"
        r"(?P<op>>=|<=|!=|=|>|<)\s*"
        r"(?P<rhs>.+?)"           # anything up to the closing paren
        r"\)\s*;?$"
    )
    PHI_RE      = re.compile(r"phi\(\s*(?P<a>\w+_\d+)\s*,\s*(?P<b>\w+_\d+)\s*\)")
    ARITH_ANY_RE = re.compile(r"\b(?P<x>\w+_\d+|\d+)\s*(?P<op>[+\-])\s*(?P<y>\w+_\d+|\d+)\b")

    def __init__(self, ssa_text: str):
        self.lines       = ssa_text.splitlines()
        self.assigns     = []   # list of (lhs, rhs_smt)
        self.cmp_asserts = []   # list of (lhs, op, rhs_smt)
        self._parse()

    def _parse(self):
        for raw in self.lines:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue

            # 1) assert(...)?
            ma = self.ASSERT_RE.match(line)
            if ma:
                lhs, op, rhs = ma.group('lhs', 'op', 'rhs')
                rhs_smt = self._rewrite_phi(rhs)
                rhs_smt = self._rewrite_arith(rhs_smt)
                self.cmp_asserts.append((lhs, op, rhs_smt))
                continue

            # 2) assignment?
            m = self.ASSIGN_RE.match(line)
            if m:
                lhs, rhs = m.group('lhs', 'rhs')
                rhs_smt = self._rewrite_phi(rhs)
                rhs_smt = self._rewrite_arith(rhs_smt)
                self.assigns.append((lhs, rhs_smt))

    def _rewrite_phi(self, expr: str) -> str:
        return self.PHI_RE.sub(r"(ite true \g<a> \g<b>)", expr)

    def _rewrite_arith(self, expr: str) -> str:
        """
        Repeatedly rewrite every `a + b` or `a - b` into SMT prefix form.
        """
        prev = None
        while prev != expr:
            prev = expr
            expr = self.ARITH_ANY_RE.sub(r"(\2 \g<x> \g<y>)", expr)
        return expr

    def to_smt2(self) -> str:
        # A) Collect only the truly used SSA names:
        decl_vars = set()

        #   - final LHS names
        for lhs, _ in self.assigns:
            decl_vars.add(lhs)

        #   - SSA names used in RHS of assignments
        for _, rhs in self.assigns:
            decl_vars.update(re.findall(r"\b\w+_\d+\b", rhs))

        #   - names in comparison asserts
        for lhs, op, rhs in self.cmp_asserts:
            decl_vars.add(lhs)
            decl_vars.update(re.findall(r"\b\w+_\d+\b", rhs))

        # B) Declare them
        decls = [f"(declare-fun {v} () Int)" for v in sorted(decl_vars)]

        # C) Build assert lines
        eqs  = [f"(assert (= {lhs} {rhs}))" for lhs, rhs in self.assigns]
        cmps = [f"(assert ({op} {lhs} {rhs}))" for lhs, op, rhs in self.cmp_asserts]

        # D) Assemble
        parts = decls + [""] + eqs
        if cmps:
            parts += [""] + cmps
        parts += ["", "(check-sat)", "(get-model)"]
        return "\n".join(parts)