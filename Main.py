from Abstract_Syntax_Tree_Gen import abstract_syntax_tree_parser
from AST_To_SSA_Transformer import SSATransformer, SSAPhiCodeGen
from SSA_To_SMT_Gen import SSAtoSMTLib

from lark import Lark, Transformer, v_args, Tree

code="""
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
    assert(arr[0] < arr[1]);
    assert(arr[1] > arr[2]);
"""
ast=abstract_syntax_tree_parser.parse(code)
print("AST")
print(ast)

ssa_intermediate_transformer=SSATransformer()
ssa_ast=ssa_intermediate_transformer.transform_block(ast.children if isinstance(ast,Tree) else ast)
print("SSA_AST")
print(ssa_ast)

ssa_gen = SSAPhiCodeGen(unroll=3)
ssa_final = ssa_gen.gen(ssa_ast)
print("SSA_FINAL")
print(ssa_final)

translator = SSAtoSMTLib(ssa_final)
smt2_output = translator.to_smt2()
print("SMT_OUTPUT")
print(smt2_output)

with open('output.smt2', 'w') as f:
    f.write(smt2_output)


