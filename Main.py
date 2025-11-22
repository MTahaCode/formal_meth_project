from Abstract_Syntax_Tree_Gen import abstract_syntax_tree_parser, prefix_variables
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
    assert(arr[1] < arr[2]);
"""
ast=abstract_syntax_tree_parser.parse(code)
ast=prefix_variables(ast, "p1_")
print(ast)

ssa_intermediate_transformer=SSATransformer()
ssa_ast=ssa_intermediate_transformer.transform_block(ast.children if isinstance(ast,Tree) else ast)
print(ssa_ast)

ssa_gen = SSAPhiCodeGen(unroll=3)
ssa_final = ssa_gen.gen(ssa_ast)
print(ssa_final)

translator = SSAtoSMTLib(ssa_final)
smt2_output = translator.to_smt2()

with open('output.smt2', 'w') as f:
    f.write(smt2_output)


# ##################################

# ast=abstract_syntax_tree_parser.parse(code)
# ast=prefix_variables(ast, "p1-")
# # print("AST")
# print("  ")
# print(ast)

# # for creating graph
# ssa_converter = SSAConverter()
# norm = ssa_converter.normalize(ast)
# # print("  ")
# # print(norm)
# unrolled = ssa_converter.unroll_loops(norm, count=2)
# # print("  ")
# # print(unrolled)
# pruned = ssa_converter.eliminate_dead_code(unrolled)
# print("  ")
# print(pruned)
# blocks, edges = ssa_converter.build_cfg(pruned)
# png_path = ssa_converter.visualize_cfg(unrolled, output_path='pre_ssa_cfg')

# # ssa_text = ssa_converter.convert_cfg_to_ssa(blocks, edges)
# # print("  ")
# # print(ssa_text)
# # print(ssa)

# # smt_translator = SSA2SMTConverter()
# # smt_translator.convert(ssa_text)
# # print(smt_translator.to_smtlib())

# ssa_intermediate_transformer=SSATransformer()
# ssa_ast=ssa_intermediate_transformer.transform_block(ast.children if isinstance(ast,Tree) else ast)
# # print(ssa_intermediate_transformer.current_version_map())
# print("SSA_AST")
# print(ssa_ast)

# ssa_gen = SSAPhiCodeGen(unroll=3)
# ssa_final = ssa_gen.gen(ssa_ast)
# # print(ssa_gen.current_version_map())
# print("SSA_FINAL")
# print(ssa_final)

# translator = SSAtoSMTLib(ssa_final)
# smt2_output = translator.to_smt2()
# # print(translator.get_latest_map())
# print("SMT_OUTPUT")
# print(smt2_output)

# with open('output.smt2', 'w') as f:
#     f.write(smt2_output)







