import re

import re

class SSAtoSMTLib:
    """
    Convert SSA-style code (as a single string) into SMT-LIB v2 format.
    Comments (lines starting with '#') are ignored.
    All variables are Int by default; infix `a + b` or `a - b` become prefix `(+ a b)`/`(- a b)`;
    `phi(a,b)` is rewritten to `(ite true a b)`.
    """
    ASSIGN_RE = re.compile(r"^(?P<lhs>\w+_\d+)\s*=\s*(?P<rhs>.+)$")
    PHI_RE    = re.compile(r"phi\(\s*(?P<a>\w+_\d+)\s*,\s*(?P<b>\w+_\d+)\s*\)")
    ARITH_RE  = re.compile(r"^(?P<x>\w+_\d+|\d+)\s*(?P<op>[+\-])\s*(?P<y>\w+_\d+|\d+)$")

    def __init__(self, ssa_text: str):
        self.lines = ssa_text.splitlines()
        self.vars = set()
        self.asserts = []
        self._parse()

    def _parse(self):
        for raw in self.lines:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            m = self.ASSIGN_RE.match(line)
            if not m:
                continue
            lhs, rhs = m.group('lhs'), m.group('rhs')
            self.vars.add(lhs)
            # rewrite phi
            rhs_smt = self._rewrite_phi(rhs)
            # rewrite simple arithmetic
            rhs_smt = self._rewrite_arith(rhs_smt)
            # record vars from rhs
            for v in re.findall(r"\b(\w+_\d+)\b", rhs_smt):
                self.vars.add(v)
            self.asserts.append((lhs, rhs_smt))

    def _rewrite_phi(self, expr: str) -> str:
        # phi(a,b) => (ite true a b)
        return self.PHI_RE.sub(r"(ite true \g<a> \g<b>)", expr)

    def _rewrite_arith(self, expr: str) -> str:
        # match a single binary op and convert to prefix
        m = self.ARITH_RE.match(expr.strip())
        if m:
            x, op, y = m.group('x', 'op', 'y')
            return f"({op} {x} {y})"
        return expr

    def to_smt2(self) -> str:
        # declare Int vars
        decls = [f"(declare-fun {v} () Int)" for v in sorted(self.vars)]
        asserts = [f"(assert (= {lhs} {rhs}))" for lhs, rhs in self.asserts]
        return "\n".join(decls + [""] + asserts + ["", "(check-sat)", "(get-model)"])