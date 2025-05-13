import streamlit as st
from Abstract_Syntax_Tree_Gen import abstract_syntax_tree_parser
from AST_To_SSA_Transformer import SSATransformer, SSAPhiCodeGen
from SSA_To_SMT_Gen import SSAtoSMTLib
from lark import Tree
from z3 import Solver, parse_smt2_string, sat, Or, And

# Normalize whitespace
def normalize_whitespace(code: str) -> str:
    return code.replace('\u00A0', ' ').replace('\u200B', ' ')

# Generate AST, SSA IR, optional C code, and SMT2
@st.cache_data
def generate_ssa_and_smt(code: str, unroll: int):
    code = normalize_whitespace(code)
    ast = abstract_syntax_tree_parser.parse(code)
    transformer = SSATransformer()
    block = ast.children if isinstance(ast, Tree) else ast
    ssa_ast = transformer.transform_block(block)
    phi = SSAPhiCodeGen(unroll)
    ssa_ir = phi.gen(ssa_ast)
    try:
        ssa_c = phi.gen_code()
    except AttributeError:
        ssa_c = None
    smt2 = SSAtoSMTLib(ssa_ir).to_smt2()
    return ast, ssa_ir, ssa_c, smt2

# Strip declare-fun lines to avoid duplicates
def strip_decls(smt: str) -> str:
    return '\n'.join(l for l in smt.splitlines() if not l.strip().startswith('(declare-fun'))

# Run Z3 SAT check, return status and up to max_models
def run_z3_models(smt: str, max_models: int = 2):
    solver = Solver()
    solver.add(parse_smt2_string(smt))
    models = []
    for _ in range(max_models):
        if solver.check() == sat:
            m = solver.model()
            models.append(m)
            block = [d() != m[d] for d in m.decls()]
            solver.add(Or(*block))
        else:
            break
    return ('SAT' if models else 'UNSAT'), models

# Single-model run (for witness)
def run_z3_single(smt: str):
    solver = Solver()
    solver.add(parse_smt2_string(smt))
    if solver.check() == sat:
        return solver.model()
    return None

# UI
st.title('Formal Methods Tool: Verification & Equivalence')
mode = st.sidebar.selectbox('Mode', ['Verification', 'Equivalence'])
unroll = st.sidebar.number_input('Unroll depth', 1, 10, 3)

# Inputs
if mode == 'Verification':
    code1 = st.text_area('Program to verify:', height=200, value="""
n := 3;
x := 4;
assert(n > x);
""")
    compare_var = None
else:
    col1, col2 = st.columns(2)
    code1 = col1.text_area('Program 1:', height=200)
    code2 = col2.text_area('Program 2:', height=200)
    compare_var = st.text_input('Output variable:', 'arr')

if st.button('Run'):
    with st.spinner('Processing...'):
        try:
            # Generate SMT for first program
            ast1, ssa1, c1, smt1 = generate_ssa_and_smt(code1, unroll)
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(['AST', 'SSA/C', 'SMT', 'Results'])
            # Show AST
            with tab1:
                st.subheader('AST Program 1')
                st.code(str(ast1))
                if mode == 'Equivalence':
                    ast2, ssa2, c2, smt2 = generate_ssa_and_smt(code2, unroll)
                    st.subheader('AST Program 2')
                    st.code(str(ast2))
            # SSA/C
            with tab2:
                st.subheader('SSA IR Program 1')
                st.code(ssa1)
                if c1:
                    st.subheader('SSA C Program 1')
                    st.code(c1)
                if mode == 'Equivalence':
                    st.subheader('SSA IR Program 2')
                    st.code(ssa2)
                    if c2:
                        st.subheader('SSA C Program 2')
                        st.code(c2)
            # SMT view
            with tab3:
                st.subheader('SMT Program 1')
                st.code(smt1)
                if mode == 'Equivalence':
                    smt2_ren = smt2.replace(f" {compare_var}", f" {compare_var}_2").replace(f"({compare_var}", f"({compare_var}_2")
                    smt2_f = strip_decls(smt2_ren)
                    st.subheader('SMT Program 2')
                    st.code(smt2_f)
            # Results
            with tab4:
                if mode == 'Verification':
                    status, models = run_z3_models(smt1)
                    if status == 'UNSAT':
                        st.success('✅ Assertion holds (no counterexamples).')
                        # produce witness by checking positive property
                        # not needed for single-program
                    else:
                        st.error(f'❌ {len(models)} counterexamples:')
                        for m in models:
                            st.json({str(d): m[d].as_long() for d in m.decls()})
                else:
                    # Equivalence: difference check
                    # build elementwise inequality
                    # derive n from smt1 declarations
                    # assume array indices 0..n-1, here hardcode 3
                    eqs = [f"(= {compare_var}_{i} {compare_var}_{i}_2)" for i in range(3)]
                    diff = f"(assert (not (and {' '.join(eqs)})))"
                    combined = smt1 + '\n' + smt2_f + '\n' + diff
                    status, models = run_z3_models(combined)
                    if status == 'UNSAT':
                        st.success('✅ Programs equivalent.')
                        # witness: all equal
                        eq = f"(assert (and {' '.join(eqs)}))"
                        witness_smt = smt1 + '\n' + smt2_f + '\n' + eq
                        w = run_z3_single(witness_smt)
                        if w:
                            st.write('Example input where outputs match:')
                            st.json({str(d): w[d].as_long() for d in w.decls()})
                    else:
                        st.error(f'❌ {len(models)} distinguishing inputs:')
                        for m in models:
                            st.json({str(d): m[d].as_long() for d in m.decls()})
        except Exception as e:
            st.exception(e)
