import streamlit as st
from lark import Lark, Transformer, v_args
import z3
import graphviz
import re
import copy

# --- 1. Mini-Language Grammar ---
grammar = r"""
    ?start: program

    program: statement*

    statement: assignment_stmt ";"
             | array_assignment_stmt ";"
             | if_stmt
             | for_stmt
             | while_stmt
             | assert_stmt ";"
             | block

    block: "{" statement* "}"

    assignment_stmt: CNAME ":=" expr
    array_assignment_stmt: CNAME "[" expr "]" ":=" expr
    
    if_stmt: "if" "(" condition ")" block ("else" block)?
    
    for_stmt: "for" "(" assignment_stmt ";" condition ";" assignment_stmt ";)" block
    while_stmt: "while" "(" condition ")" block
    
    assert_stmt: "assert" "(" condition ")"

    ?condition: expr // Conditions are expressions that evaluate to bool-like (non-zero for true)

    ?expr: term (("+" | "-") term)*
         | array_literal

    ?term: factor (("*" | "/" | "%") factor)*

    ?factor: NUMBER
           | CNAME "[" expr "]" -> array_lookup
           | CNAME
           | "(" expr ")"
           | ("!"|"-" ) factor -> unary_op // Added unary negation and logical not


    array_literal: "[" [expr ("," expr)*] "]"

    %import common.CNAME
    %import common.NUMBER
    %import common.WS
    %ignore WS
"""

# --- 2. AST Transformer ---
class AstTransformer(Transformer):
    @v_args(inline=True)
    def NUMBER(self, value):
        return ("number", int(value))

    @v_args(inline=True)
    def CNAME(self, value):
        return ("var", str(value))

    def array_literal(self, items):
        return ("array_literal", [item for item in items if item is not None]) # Filter out None if list is empty

    @v_args(inline=True)
    def array_lookup(self, arr_name, index_expr):
        return ("array_lookup", arr_name, index_expr)

    def assignment_stmt(self, args):
        return ("assign", args[0], args[1])

    def array_assignment_stmt(self, args):
        return ("array_assign", args[0], args[1], args[2])

    def if_stmt(self, args):
        cond = args[0]
        then_block = args[1]
        else_block = args[2] if len(args) > 2 else ("block", [])
        return ("if", cond, then_block, else_block)

    def for_stmt(self, args):
        init, cond, update, body = args
        return ("for", init, cond, update, body)
    
    def while_stmt(self, args):
        cond, body = args
        return ("while", cond, body)

    def assert_stmt(self, args):
        return ("assert", args[0])

    def block(self, stmts):
        return ("block", list(stmts))

    def program(self, stmts):
        return ("program", list(stmts))

    def expr(self, args):
        if len(args) == 1:
            return args[0]
        op = args[1]
        if op == '+':
            return ("binop", "+", args[0], args[2])
        elif op == '-':
            return ("binop", "-", args[0], args[2])
        else: # Handle boolean ops
            return ("binop", op, args[0], args[2])


    def term(self, args):
        if len(args) == 1:
            return args[0]
        op = args[1]
        return ("binop", op, args[0], args[2])

    def factor(self, args):
        return args[0] # Already structured by rules like array_lookup, CNAME, NUMBER

    def condition(self, args): # Conditions are expressions
        if len(args) == 1: # Simple condition
            return args[0]
        # This handles comparisons like x < y, x == y etc.
        # Assuming comparison operators are defined in expr/term/factor
        # For simplicity, we'll assume expr handles this.
        # If we need explicit bool ops:
        # e.g. expr ("<" | ">" | "==" | "!=" | "<=" | ">=") expr
        # Then this method would create a ("comparison_op", op, left, right) tuple
        return args[0] # For now, conditions are general expressions.

    def unary_op(self, args):
        op, operand = args
        return ("unary_op", str(op), operand)


# Initialize parser
parser = Lark(grammar, parser='lalr', transformer=AstTransformer())

# --- 3. Loop Unroller ---
def unroll_loops_in_ast(ast_node, unroll_depth):
    if isinstance(ast_node, tuple):
        op = ast_node[0]
        args = ast_node[1:]
        
        if op == "for":
            init_stmt, cond_expr, update_stmt, body_block = args
            unrolled_stmts = [init_stmt]
            
            # Create unique loop vars for unrolling if they are simple assignments
            # This is a simplification; full alpha-conversion is more complex
            loop_var_name = None
            if init_stmt[0] == 'assign':
                loop_var_name = init_stmt[1][1] # ('var', 'i') -> 'i'
            
            original_loop_var_name = loop_var_name

            for i in range(unroll_depth):
                # Create a copy of the body and condition for this iteration
                current_body = copy.deepcopy(body_block)
                current_cond = copy.deepcopy(cond_expr)

                # If loop var used in condition or body, it should be the latest version
                # This renaming is a bit naive and might need refinement for complex cases
                if loop_var_name:
                    # Rename loop var in condition and body for this iteration
                    # This is a simplistic renaming. A proper renaming would traverse the AST.
                    def rename_in_ast(node, old_name, new_name):
                        if isinstance(node, tuple):
                            if node[0] == 'var' and node[1] == old_name:
                                return ('var', new_name)
                            return (node[0], *[rename_in_ast(arg, old_name, new_name) for arg in node[1:]])
                        elif isinstance(node, list):
                            return [rename_in_ast(item, old_name, new_name) for item in node]
                        return node
                    
                    # For the *first* iteration, the init var is used.
                    # For subsequent, the updated var from previous iteration is used.
                    # This renaming logic needs to be context-aware. For now, we rely on SSA to handle versions.
                    # The unroller primarily structures the repetition.

                # The condition guards this iteration's body and update
                # if (condition) { body; update; }
                iteration_block_stmts = []
                if current_body[0] == 'block':
                    iteration_block_stmts.extend(current_body[1])
                else: # single statement
                    iteration_block_stmts.append(current_body)
                
                iteration_block_stmts.append(copy.deepcopy(update_stmt)) # Update at the end of iteration body
                
                unrolled_stmts.append(
                    ("if", current_cond, ("block", iteration_block_stmts), ("block", []))
                )
            return ("block", unrolled_stmts)

        elif op == "while":
            cond_expr, body_block = args
            unrolled_stmts = []
            for _ in range(unroll_depth):
                # if (condition) { body; }
                iteration_block_stmts = []
                if body_block[0] == 'block':
                     iteration_block_stmts.extend(body_block[1])
                else:
                    iteration_block_stmts.append(body_block)

                unrolled_stmts.append(
                    ("if", copy.deepcopy(cond_expr), ("block", iteration_block_stmts), ("block", []))
                )
            return ("block", unrolled_stmts)
        else:
            return (op, *[unroll_loops_in_ast(arg, unroll_depth) for arg in args])
    elif isinstance(ast_node, list):
        return [unroll_loops_in_ast(item, unroll_depth) for item in ast_node]
    else:
        return ast_node


# --- 4. SSA Conversion ---
class SSATransformer:
    def __init__(self):
        self.var_versions = {}  # var_name -> current_version_index
        self.array_versions = {} # array_name -> current_version_index
        self.ssa_statements = []
        self.block_id_counter = 0 # For CFG node naming

    def _get_ssa_var(self, var_name):
        if var_name not in self.var_versions:
            self.var_versions[var_name] = 0
            return f"{var_name}_0"
        return f"{var_name}_{self.var_versions[var_name]}"

    def _new_ssa_var(self, var_name):
        self.var_versions[var_name] = self.var_versions.get(var_name, -1) + 1
        return f"{var_name}_{self.var_versions[var_name]}"

    def _get_ssa_array(self, arr_name):
        if arr_name not in self.array_versions:
            self.array_versions[arr_name] = 0
            return f"{arr_name}_0"
        return f"{arr_name}_{self.array_versions[arr_name]}"

    def _new_ssa_array(self, arr_name):
        self.array_versions[arr_name] = self.array_versions.get(arr_name, -1) + 1
        return f"{arr_name}_{self.array_versions[arr_name]}"

    def _transform_expr(self, expr_node):
        if isinstance(expr_node, tuple):
            op = expr_node[0]
            if op == "number":
                return expr_node
            elif op == "var":
                return ("var", self._get_ssa_var(expr_node[1]))
            elif op == "array_lookup":
                arr_name_node, index_expr = expr_node[1], expr_node[2] # arr_name_node is ('var', name)
                arr_ssa_name = self._get_ssa_array(arr_name_node[1])
                return ("array_lookup", ("var", arr_ssa_name), self._transform_expr(index_expr))
            elif op == "binop":
                return ("binop", expr_node[1], self._transform_expr(expr_node[2]), self._transform_expr(expr_node[3]))
            elif op == "unary_op":
                return ("unary_op", expr_node[1], self._transform_expr(expr_node[2]))
            elif op == "array_literal":
                # This will be handled by assignment to create initial array version
                return ("array_literal", [self._transform_expr(e) for e in expr_node[1]])
            else:
                raise ValueError(f"Unknown expression node type: {op}")
        return expr_node # Should be number

    def transform(self, ast_node):
        self.var_versions = {}
        self.array_versions = {}
        self.ssa_statements = []
        self._transform_node(ast_node)
        return self.ssa_statements

    def _transform_node(self, node):
        if not isinstance(node, tuple): # Should not happen with parsed AST
            return

        op = node[0]
        args = node[1:]

        if op == "program":
            for stmt in args[0]:
                self._transform_node(stmt)
        elif op == "block":
            for stmt in args[0]:
                self._transform_node(stmt)
        elif op == "assign":
            var_node, expr_node = args
            var_name = var_node[1]
            
            # Handle array literal assignment
            if expr_node[0] == "array_literal":
                new_arr_ssa_name = self._new_ssa_array(var_name)
                transformed_elements = [self._transform_expr(e) for e in expr_node[1]]
                self.ssa_statements.append(("array_init", ("var", new_arr_ssa_name), transformed_elements))
            else: # Scalar assignment
                new_var_ssa_name = self._new_ssa_var(var_name)
                transformed_expr = self._transform_expr(expr_node)
                self.ssa_statements.append(("assign", ("var", new_var_ssa_name), transformed_expr))

        elif op == "array_assign":
            arr_node, index_expr, value_expr = args
            arr_name = arr_node[1] # ('var', name) -> name
            
            old_arr_ssa_name = self._get_ssa_array(arr_name)
            new_arr_ssa_name = self._new_ssa_array(arr_name)
            
            transformed_index = self._transform_expr(index_expr)
            transformed_value = self._transform_expr(value_expr)
            
            self.ssa_statements.append(("array_assign", 
                                        ("var", new_arr_ssa_name), 
                                        ("var", old_arr_ssa_name), 
                                        transformed_index, 
                                        transformed_value))
        elif op == "if":
            cond_expr, then_block, else_block = args
            transformed_cond = self._transform_expr(cond_expr)
            
            # Save current versions before branching
            vars_before_if = self.var_versions.copy()
            arrays_before_if = self.array_versions.copy()
            
            ssa_then_branch = []
            # Process then branch
            # Create a temporary SSATransformer for the branch to isolate version changes
            then_transformer = SSATransformer()
            then_transformer.var_versions = self.var_versions.copy() # Pass current state
            then_transformer.array_versions = self.array_versions.copy()
            then_transformer._transform_node(then_block)
            ssa_then_branch = then_transformer.ssa_statements
            vars_after_then = then_transformer.var_versions
            arrays_after_then = then_transformer.array_versions

            # Restore versions for else branch from before the if
            self.var_versions = vars_before_if.copy()
            self.array_versions = arrays_before_if.copy()

            ssa_else_branch = []
            # Process else branch
            else_transformer = SSATransformer()
            else_transformer.var_versions = self.var_versions.copy() # Pass current state (restored)
            else_transformer.array_versions = self.array_versions.copy()
            else_transformer._transform_node(else_block)
            ssa_else_branch = else_transformer.ssa_statements
            vars_after_else = else_transformer.var_versions
            arrays_after_else = else_transformer.array_versions
            
            self.ssa_statements.append(("if", transformed_cond, ssa_then_branch, ssa_else_branch))

            # Phi-like merge: For any variable modified in either branch, create a new version
            # The SMT translation will use ITE for this.
            # Here, we update the main version counters to the max of branches.
            # This is a simplification. True SSA needs explicit Phi nodes.
            
            merged_vars = set(vars_after_then.keys()) | set(vars_after_else.keys())
            for var in merged_vars:
                # SMT will handle this with (ite cond then_var_version else_var_version)
                # For SSA text, we might just state the versions from each branch.
                # The actual "phi" happens implicitly in SMT.
                # We need to ensure the main var_versions reflects that a var *might* have changed.
                # So, we advance its version if it changed in *either* branch relative to before_if.
                
                # Get the version of var *before* the if
                v_before = vars_before_if.get(var, -1)
                v_then = vars_after_then.get(var, v_before)
                v_else = vars_after_else.get(var, v_before)

                if v_then != v_before or v_else != v_before: # If changed in any branch
                    # Create a new "merged" version
                    new_merged_version_name = self._new_ssa_var(var)
                    
                    # Get the SSA names for the versions from branches
                    then_var_ssa = f"{var}_{v_then}" if v_then != -1 else None
                    else_var_ssa = f"{var}_{v_else}" if v_else != -1 else None
                    
                    # If var wasn't defined before if but defined in a branch
                    # (This case should be handled if var_versions start from -1 or similar)
                    # For now, assume vars used in phi were defined before or in both branches

                    # Find the source for the else branch value if it wasn't modified in else
                    if v_else == v_before and else_var_ssa: # it was defined before, and not changed in else
                        pass # else_var_ssa is already correct (e.g., x_0)
                    elif not else_var_ssa and v_before != -1 : # not defined in else, but was before
                         else_var_ssa = f"{var}_{v_before}"
                    
                    # Similar for then
                    if v_then == v_before and then_var_ssa:
                        pass
                    elif not then_var_ssa and v_before != -1:
                        then_var_ssa = f"{var}_{v_before}"

                    # The actual phi assignment in SSA text form
                    self.ssa_statements.append(
                        ("phi_assign", 
                         ("var", new_merged_version_name), 
                         transformed_cond, 
                         ("var", then_var_ssa) if then_var_ssa else None, # Can be None if var only defined in one branch
                         ("var", else_var_ssa) if else_var_ssa else None
                        )
                    )
                else: # Not changed in either branch, restore original version count
                    self.var_versions[var] = v_before


            merged_arrays = set(arrays_after_then.keys()) | set(arrays_after_else.keys)
            for arr in merged_arrays:
                a_before = arrays_before_if.get(arr, -1)
                a_then = arrays_after_then.get(arr, a_before)
                a_else = arrays_after_else.get(arr, a_before)

                if a_then != a_before or a_else != a_before:
                    new_merged_arr_version_name = self._new_ssa_array(arr)
                    then_arr_ssa = f"{arr}_{a_then}" if a_then != -1 else None
                    else_arr_ssa = f"{arr}_{a_else}" if a_else != -1 else None

                    if not else_arr_ssa and a_before != -1:
                        else_arr_ssa = f"{arr}_{a_before}"
                    if not then_arr_ssa and a_before != -1:
                        then_arr_ssa = f"{arr}_{a_before}"
                    
                    self.ssa_statements.append(
                        ("phi_assign_array",
                         ("var", new_merged_arr_version_name),
                         transformed_cond,
                         ("var", then_arr_ssa) if then_arr_ssa else None,
                         ("var", else_arr_ssa) if else_arr_ssa else None
                        )
                    )
                else:
                    self.array_versions[arr] = a_before


        elif op == "assert":
            cond_expr = args[0]
            transformed_cond = self._transform_expr(cond_expr)
            self.ssa_statements.append(("assert", transformed_cond))
        
        # For and While loops are expected to be unrolled before SSA
        elif op == "for" or op == "while":
            st.error("Loops should be unrolled before SSA conversion. This indicates an issue.")
            # This part should ideally not be reached if unrolling is done first.
            # If it is, it means the unroller didn't handle it, or SSA was called on non-unrolled AST.
            # For simplicity, we'll assume unrolling precedes SSA.


# --- Helper to pretty-print SSA ---
def pretty_print_ssa_expr(ssa_expr):
    if not isinstance(ssa_expr, tuple):
        return str(ssa_expr)
    op = ssa_expr[0]
    if op == "number": return str(ssa_expr[1])
    if op == "var": return ssa_expr[1]
    if op == "array_lookup":
        # ssa_expr[1] is ('var', arr_ssa_name)
        return f"{ssa_expr[1][1]}[{pretty_print_ssa_expr(ssa_expr[2])}]"
    if op == "binop":
        op_symbol = ssa_expr[1]
        # Map SMT op names to nicer symbols if needed
        readable_ops = {"bvadd": "+", "bvsub": "-", "bvmul": "*", "bvsdiv": "/", "bvsrem": "%",
                        "bvslt": "<", "bvsle": "<=", "bvsgt": ">", "bvsge": ">=",
                        "=": "==", "distinct": "!="} # Z3 uses distinct for !=
        op_symbol = readable_ops.get(op_symbol, op_symbol)
        return f"({pretty_print_ssa_expr(ssa_expr[2])} {op_symbol} {pretty_print_ssa_expr(ssa_expr[3])})"
    if op == "unary_op":
        op_symbol = ssa_expr[1]
        if op_symbol == '!': op_symbol = "not " # for logical not
        return f"{op_symbol}{pretty_print_ssa_expr(ssa_expr[2])}"
    if op == "array_literal":
        return f"[{', '.join(pretty_print_ssa_expr(e) for e in ssa_expr[1])}]"
    return str(ssa_expr)


def pretty_print_ssa_stmts(ssa_stmts, indent=0):
    output = []
    indent_str = "  " * indent
    for stmt in ssa_stmts:
        op = stmt[0]
        if op == "assign":
            # stmt[1] is ('var', var_ssa_name)
            output.append(f"{indent_str}{stmt[1][1]} := {pretty_print_ssa_expr(stmt[2])}")
        elif op == "array_init":
            # stmt[1] is ('var', arr_ssa_name)
            elements_str = ", ".join(pretty_print_ssa_expr(e) for e in stmt[2])
            output.append(f"{indent_str}{stmt[1][1]} := [{elements_str}]")
        elif op == "array_assign":
            # stmt[1] is new_arr_name ('var', name), stmt[2] is old_arr_name ('var', name)
            output.append(f"{indent_str}{stmt[1][1]} := Store({stmt[2][1]}, {pretty_print_ssa_expr(stmt[3])}, {pretty_print_ssa_expr(stmt[4])})")
        elif op == "if":
            cond_str = pretty_print_ssa_expr(stmt[1])
            output.append(f"{indent_str}if ({cond_str}) {{")
            output.extend(pretty_print_ssa_stmts(stmt[2], indent + 1))
            if stmt[3]: # Else branch exists and has statements
                output.append(f"{indent_str}}} else {{")
                output.extend(pretty_print_ssa_stmts(stmt[3], indent + 1))
            output.append(f"{indent_str}}}")
        elif op == "phi_assign":
            # target_var is ('var', name)
            target_var_name = stmt[1][1]
            cond_str = pretty_print_ssa_expr(stmt[2])
            then_val_str = pretty_print_ssa_expr(stmt[3]) if stmt[3] else "undef_then"
            else_val_str = pretty_print_ssa_expr(stmt[4]) if stmt[4] else "undef_else"
            output.append(f"{indent_str}{target_var_name} := Phi({cond_str} ? {then_val_str} : {else_val_str})")
        elif op == "phi_assign_array":
            target_arr_name = stmt[1][1]
            cond_str = pretty_print_ssa_expr(stmt[2])
            then_arr_str = pretty_print_ssa_expr(stmt[3]) if stmt[3] else "undef_then_arr"
            else_arr_str = pretty_print_ssa_expr(stmt[4]) if stmt[4] else "undef_else_arr"
            output.append(f"{indent_str}{target_arr_name} := Phi_Array({cond_str} ? {then_arr_str} : {else_arr_str})")

        elif op == "assert":
            output.append(f"{indent_str}assert({pretty_print_ssa_expr(stmt[1])})")
    return output


# --- 5. SMT Constraint Generation ---
class SMTGenerator:
    def __init__(self, unroll_depth_for_array_size_heuristic=5):
        self.solver = z3.Solver()
        self.z3_vars = {}  # SSA_var_name -> z3_var
        self.z3_arrays = {} # SSA_arr_name -> z3_array_var
        self.smt_constraints_str = []
        self.constraints = [] # Z3 constraints
        # Heuristic for array bounds, can be improved or made configurable
        self.max_array_elements = unroll_depth_for_array_size_heuristic * 2 + 10 # Max elements for array init

    def _to_z3_expr(self, ssa_expr, current_path_condition=z3.BoolVal(True)):
        if not isinstance(ssa_expr, tuple): # literal number
            return z3.IntVal(ssa_expr) 
        
        op = ssa_expr[0]
        if op == "number":
            return z3.IntVal(ssa_expr[1])
        elif op == "var":
            var_name = ssa_expr[1]
            if var_name not in self.z3_vars:
                self.z3_vars[var_name] = z3.Int(var_name)
                self.smt_constraints_str.append(f"(declare-const {var_name} Int)")
            return self.z3_vars[var_name]
        elif op == "array_lookup":
            # ssa_expr[1] is ('var', arr_ssa_name)
            arr_ssa_name = ssa_expr[1][1]
            if arr_ssa_name not in self.z3_arrays:
                 # Default to Int Int arrays. Could be configurable.
                self.z3_arrays[arr_ssa_name] = z3.Array(arr_ssa_name, z3.IntSort(), z3.IntSort())
                self.smt_constraints_str.append(f"(declare-const {arr_ssa_name} (Array Int Int))")

            z3_arr = self.z3_arrays[arr_ssa_name]
            z3_index = self._to_z3_expr(ssa_expr[2], current_path_condition)
            return z3.Select(z3_arr, z3_index)
        elif op == "binop":
            op_str = ssa_expr[1]
            left = self._to_z3_expr(ssa_expr[2], current_path_condition)
            right = self._to_z3_expr(ssa_expr[3], current_path_condition)
            if op_str == "+": return left + right
            if op_str == "-": return left - right
            if op_str == "*": return left * right
            if op_str == "/": return left / right # Integer division
            if op_str == "%": return left % right
            # Comparisons - these evaluate to Z3 BoolRef
            if op_str == "<": return left < right
            if op_str == "<=": return left <= right
            if op_str == ">": return left > right
            if op_str == ">=": return left >= right
            if op_str == "==": return left == right # Note: Lark rule might need "=="
            if op_str == "!=": return left != right
            # Logical ops (assuming expr can be bool from comparisons)
            # These are tricky because mini-language doesn't have explicit bool type
            # We assume 0 is false, non-zero is true for conditions in 'if'
            # For SMT 'assert', the condition should directly map to Z3 Bool.
            if op_str == "&&": return z3.And(left != 0, right != 0) # Example mapping
            if op_str == "||": return z3.Or(left != 0, right != 0)  # Example mapping
            raise ValueError(f"Unknown SMT binary operator: {op_str}")
        elif op == "unary_op":
            op_str = ssa_expr[1]
            operand = self._to_z3_expr(ssa_expr[2], current_path_condition)
            if op_str == "-": return -operand # Arithmetic negation
            if op_str == "!": return z3.Not(operand != 0) # Logical not (non-zero is true)
            raise ValueError(f"Unknown SMT unary operator: {op_str}")
        else: # array_literal should not reach here directly, handled by array_init
            raise ValueError(f"Cannot convert SSA expression to Z3: {ssa_expr}")


    def _process_ssa_stmt(self, ssa_stmt, current_path_condition=z3.BoolVal(True)):
        op = ssa_stmt[0]
        if op == "assign":
            # ssa_stmt[1] is ('var', target_ssa_name)
            target_var_name = ssa_stmt[1][1]
            z3_target_var = self._to_z3_expr(("var", target_var_name), current_path_condition) # Ensures declaration
            z3_expr_val = self._to_z3_expr(ssa_stmt[2], current_path_condition)
            constraint = (z3_target_var == z3_expr_val)
            self.constraints.append(z3.Implies(current_path_condition, constraint))
            self.smt_constraints_str.append(f"(assert (=> {self.solver.to_smt2(current_path_condition)} {self.solver.to_smt2(constraint)}))")

        elif op == "array_init":
            # ssa_stmt[1] is ('var', arr_ssa_name)
            arr_ssa_name = ssa_stmt[1][1]
            elements = ssa_stmt[2]

            if arr_ssa_name not in self.z3_arrays:
                self.z3_arrays[arr_ssa_name] = z3.Array(arr_ssa_name, z3.IntSort(), z3.IntSort())
                self.smt_constraints_str.append(f"(declare-const {arr_ssa_name} (Array Int Int))")
            
            z3_arr = self.z3_arrays[arr_ssa_name]
            
            # Create a base array (e.g., initialized to 0 or a symbolic const)
            # For simplicity, let's try to build it with stores.
            # Z3 does not have direct array literal syntax like {1, 2, 3}
            # We model arr := [e1, e2, ...] as:
            # arr_temp0 = Store(const_array, 0, e1)
            # arr_temp1 = Store(arr_temp0, 1, e2) ...
            # arr = arr_tempN
            
            # For bounded verification, we might declare K arrays for K elements.
            # Here, we build it up with Stores.
            # A default value for uninitialized parts of the array.
            # This is complex. A simpler way for fixed init:
            # assert Select(arr, 0) == e0, Select(arr, 1) == e1, ...
            
            # Initialize with stores from a symbolic "empty" array
            # To avoid issues with `Store` chains on the *same* array variable if not careful
            # create a distinct z3 array for this initialization step
            current_arr_val = z3.K(z3.IntSort(), z3.IntVal(0)) # Default to 0 for elements not explicitly set

            for i, elem_expr in enumerate(elements):
                z3_elem_val = self._to_z3_expr(elem_expr, current_path_condition)
                current_arr_val = z3.Store(current_arr_val, z3.IntVal(i), z3_elem_val)

            constraint = (z3_arr == current_arr_val)
            self.constraints.append(z3.Implies(current_path_condition, constraint))
            self.smt_constraints_str.append(f"(assert (=> {self.solver.to_smt2(current_path_condition)} {self.solver.to_smt2(constraint)}))")


        elif op == "array_assign":
            # new_arr_ssa ('var', name), old_arr_ssa ('var', name), index_expr, val_expr
            new_arr_ssa_name = ssa_stmt[1][1]
            old_arr_ssa_name = ssa_stmt[2][1]

            z3_new_arr = self._to_z3_expr(("var", new_arr_ssa_name), current_path_condition) # Declares if new
            z3_old_arr = self._to_z3_expr(("var", old_arr_ssa_name), current_path_condition) # Declares if new (should exist)

            z3_index = self._to_z3_expr(ssa_stmt[3], current_path_condition)
            z3_value = self._to_z3_expr(ssa_stmt[4], current_path_condition)
            
            constraint = (z3_new_arr == z3.Store(z3_old_arr, z3_index, z3_value))
            self.constraints.append(z3.Implies(current_path_condition, constraint))
            self.smt_constraints_str.append(f"(assert (=> {self.solver.to_smt2(current_path_condition)} {self.solver.to_smt2(constraint)}))")

        elif op == "if":
            cond_expr, then_branch_ssa, else_branch_ssa = ssa_stmt[1], ssa_stmt[2], ssa_stmt[3]
            z3_cond = self._to_z3_expr(cond_expr, current_path_condition)
            
            # The condition for Z3 needs to be a BoolRef.
            # Our mini-language `if(expr)` implies `if(expr != 0)`.
            if not isinstance(z3_cond, z3.BoolRef):
                z3_cond = (z3_cond != 0)

            # Process then branch under path condition: current_path_condition AND z3_cond
            for stmt in then_branch_ssa:
                self._process_ssa_stmt(stmt, z3.And(current_path_condition, z3_cond))
            
            # Process else branch under path condition: current_path_condition AND (NOT z3_cond)
            for stmt in else_branch_ssa:
                self._process_ssa_stmt(stmt, z3.And(current_path_condition, z3.Not(z3_cond)))
        
        elif op == "phi_assign":
            # target_var ('var', name), cond_expr, then_var ('var', name), else_var ('var', name)
            target_ssa_name = ssa_stmt[1][1]
            cond_expr = ssa_stmt[2]
            then_var_expr = ssa_stmt[3] # This is ('var', ssa_name_then) or None
            else_var_expr = ssa_stmt[4] # This is ('var', ssa_name_else) or None

            z3_target_var = self._to_z3_expr(("var", target_ssa_name), current_path_condition) # Declares target

            z3_cond = self._to_z3_expr(cond_expr, current_path_condition)
            if not isinstance(z3_cond, z3.BoolRef): # Ensure it's a boolean condition for ITE
                z3_cond = (z3_cond != 0)

            # Handle cases where a variable might not be defined in a branch (e.g. assigned only in 'then')
            # In such cases, its value would be the one from *before* the if.
            # This should be implicitly handled if ssa_name_else/then correctly refer to pre-if version.
            # For SMT, if a var like `x_else` is not available, it implies `x` wasn't changed in else.
            # The SSA transformation should ensure `then_var_expr` and `else_var_expr` are valid.
            # If None, it means the variable was not defined on that path *within the if*.
            # The SMT generator must use the version of the variable *before* the if block for that path.
            # This is tricky. The current SSA transform tries to make this explicit.

            if then_var_expr is None or else_var_expr is None:
                # This means the variable was not modified on one of the paths.
                # The SMT for ITE requires both branches. The SSA needs to provide the correct var name from before.
                # This is a simplification in current SSA: assumes phi always has two source versions.
                # A robust SSA would trace back. For now, we require valid names.
                st.warning(f"Phi for {target_ssa_name} has a missing branch source. SMT might be incorrect.")
                # As a fallback, could try to find the "latest" version of the base var name before this phi.
                # This is complex. Assume SSA provides valid inputs for now.
                # If one branch is None, we can't form a simple ITE.
                # This points to a need for more robust SSA Phi node generation.
                # For now, if a branch is None, we might skip this phi or use a default.
                # Let's assume SSA provides *some* var, even if it's the pre-if version.
                if then_var_expr is None and else_var_expr is None: # Should not happen if var was defined before if
                    return # Cannot create ITE
                elif then_var_expr is None: # Use else var for both sides of ITE (effectively if !cond then else else else)
                    # This is likely wrong. Should be: if cond then var_before_if else var_from_else
                    # The SSA needs to provide var_before_if as the then_var_expr.
                    st.error(f"Phi for {target_ssa_name}: then_var_expr is None. SMT will be flawed.")
                    return # Avoid generating faulty SMT
                elif else_var_expr is None:
                    st.error(f"Phi for {target_ssa_name}: else_var_expr is None. SMT will be flawed.")
                    return


            z3_then_val = self._to_z3_expr(then_var_expr, current_path_condition)
            z3_else_val = self._to_z3_expr(else_var_expr, current_path_condition)

            constraint = (z3_target_var == z3.If(z3_cond, z3_then_val, z3_else_val))
            self.constraints.append(z3.Implies(current_path_condition, constraint))
            self.smt_constraints_str.append(f"(assert (=> {self.solver.to_smt2(current_path_condition)} {self.solver.to_smt2(constraint)}))")

        elif op == "phi_assign_array":
            target_arr_ssa_name = ssa_stmt[1][1]
            cond_expr = ssa_stmt[2]
            then_arr_expr = ssa_stmt[3] # ('var', arr_ssa_name_then) or None
            else_arr_expr = ssa_stmt[4] # ('var', arr_ssa_name_else) or None

            z3_target_arr = self._to_z3_expr(("var", target_arr_ssa_name), current_path_condition) # Declares target array

            z3_cond = self._to_z3_expr(cond_expr, current_path_condition)
            if not isinstance(z3_cond, z3.BoolRef):
                z3_cond = (z3_cond != 0)

            if then_arr_expr is None or else_arr_expr is None:
                st.error(f"Phi_Array for {target_arr_ssa_name}: a branch source is None. SMT will be flawed.")
                return
            
            z3_then_arr = self._to_z3_expr(then_arr_expr, current_path_condition)
            z3_else_arr = self._to_z3_expr(else_arr_expr, current_path_condition)
            
            constraint = (z3_target_arr == z3.If(z3_cond, z3_then_arr, z3_else_arr)) # Z3 handles ITE for arrays
            self.constraints.append(z3.Implies(current_path_condition, constraint))
            self.smt_constraints_str.append(f"(assert (=> {self.solver.to_smt2(current_path_condition)} {self.solver.to_smt2(constraint)}))")


        elif op == "assert":
            cond_expr = ssa_stmt[1]
            z3_cond = self._to_z3_expr(cond_expr, current_path_condition)
            if not isinstance(z3_cond, z3.BoolRef): # Ensure it's a boolean condition
                z3_cond = (z3_cond != 0)
            
            # For verification, this is the condition we want to check.
            # It's stored differently.
            self.program_assertions.append(z3.Implies(current_path_condition, z3_cond))
            self.smt_constraints_str.append(f"; Program Assertion under path {self.solver.to_smt2(current_path_condition)}:\n"
                                            f"; (assert {self.solver.to_smt2(z3_cond)})")


    def generate_smt(self, ssa_program):
        self.solver.reset()
        self.z3_vars = {}
        self.z3_arrays = {}
        self.smt_constraints_str = []
        self.constraints = []
        self.program_assertions = [] # Store program's assertions separately

        for ssa_stmt in ssa_program:
            self._process_ssa_stmt(ssa_stmt) # Process with top-level path condition True
        
        self.solver.add(self.constraints)
        return self.solver, self.program_assertions, "\n".join(self.smt_constraints_str), self.z3_vars, self.z3_arrays
    
    def get_final_ssa_vars(self, ssa_converter_instance):
        """Gets the final SSA version names for all variables and arrays."""
        final_vars = {}
        for var_name, version_idx in ssa_converter_instance.var_versions.items():
            final_vars[var_name] = f"{var_name}_{version_idx}"
        
        final_arrays = {}
        for arr_name, version_idx in ssa_converter_instance.array_versions.items():
            final_arrays[arr_name] = f"{arr_name}_{version_idx}"
        return final_vars, final_arrays

# --- 6. CFG Generation (Basic) ---
def generate_cfg(ast_or_ssa_stmts, is_ssa=False):
    dot = graphviz.Digraph()
    node_counter = 0
    
    def new_node_id():
        nonlocal node_counter
        node_counter += 1
        return f"N{node_counter}"

    def format_stmt_for_cfg(stmt):
        if is_ssa:
            # Use pretty_print_ssa_stmts logic for a single statement
            # This is a bit hacky; ideally, have a dedicated single SSA stmt formatter
            return pretty_print_ssa_stmts([stmt])[0].strip() if stmt else "Empty"
        else: # AST node
            if not isinstance(stmt, tuple): return str(stmt)
            # Simple formatting for AST nodes
            if stmt[0] == "assign": return f"{stmt[1][1]} := {format_expr_for_cfg(stmt[2])}"
            if stmt[0] == "array_assign": return f"{stmt[1][1]}[{format_expr_for_cfg(stmt[2])}] := {format_expr_for_cfg(stmt[3])}"
            if stmt[0] == "assert": return f"assert({format_expr_for_cfg(stmt[1])})"
            # For control flow, we just label the node type
            if stmt[0] in ["if", "for", "while"]: return stmt[0].upper()
            return str(stmt)
            
    def format_expr_for_cfg(expr):
        if not isinstance(expr, tuple): return str(expr)
        op = expr[0]
        if op == "number": return str(expr[1])
        if op == "var": return expr[1]
        if op == "array_lookup": return f"{expr[1][1]}[{format_expr_for_cfg(expr[2])}]"
        if op == "binop": return f"({format_expr_for_cfg(expr[2])} {expr[1]} {format_expr_for_cfg(expr[3])})"
        if op == "unary_op": return f"{expr[1]}{format_expr_for_cfg(expr[2])}"
        if op == "array_literal": return f"[{', '.join(format_expr_for_cfg(e) for e in expr[1])}]"
        return str(expr)

    def build_cfg_nodes(statements, parent_node_id=None, entry_edge_label=""):
        nonlocal node_counter
        current_block_stmts = []
        prev_node_id = parent_node_id

        # Group sequential statements into basic blocks
        idx = 0
        while idx < len(statements):
            stmt = statements[idx]
            
            if is_ssa: # SSA statements list
                stmt_op = stmt[0]
                is_control_flow = stmt_op in ["if", "phi_assign", "phi_assign_array"] # SSA if/phi are control markers
            else: # AST nodes list
                stmt_op = stmt[0] if isinstance(stmt, tuple) else None
                is_control_flow = stmt_op in ["if", "for", "while"]

            if not is_control_flow:
                current_block_stmts.append(stmt)
                idx += 1
            else: # Control flow statement or end of list
                if current_block_stmts:
                    # Create node for accumulated basic block
                    block_node_id = new_node_id()
                    label = "\n".join(format_stmt_for_cfg(s) for s in current_block_stmts)
                    dot.node(block_node_id, label=label, shape="box")
                    if prev_node_id:
                        dot.edge(prev_node_id, block_node_id, label=entry_edge_label)
                    prev_node_id = block_node_id
                    current_block_stmts = []
                    entry_edge_label = "" # Reset edge label after use

                # Handle the control flow statement
                control_node_id = new_node_id()
                dot.node(control_node_id, label=format_stmt_for_cfg(stmt), shape="diamond")
                if prev_node_id:
                    dot.edge(prev_node_id, control_node_id, label=entry_edge_label)
                
                if stmt_op == "if":
                    if is_ssa:
                        # stmt = ("if", cond, then_branch_ssa, else_branch_ssa)
                        # Recurse for then branch
                        build_cfg_nodes(stmt[2], control_node_id, "True")
                        # Recurse for else branch
                        if stmt[3]: # Else branch exists
                             build_cfg_nodes(stmt[3], control_node_id, "False")
                    else:
                        # stmt = ("if", cond, then_block, else_block)
                        # then_block is ("block", list_of_stmts)
                        build_cfg_nodes(stmt[2][1], control_node_id, "True")
                        if stmt[3] and stmt[3][1]: # Else block exists and is not empty
                            build_cfg_nodes(stmt[3][1], control_node_id, "False")
                # For/While for original AST (SSA should be unrolled)
                elif not is_ssa and stmt_op in ["for", "while"]:
                    # For stmt: ("for", init, cond, update, body)
                    # While stmt: ("while", cond, body)
                    body_stmts = stmt[4][1] if stmt_op == "for" else stmt[2][1]
                    # Edge to body
                    body_entry_node_id = build_cfg_nodes(body_stmts, control_node_id, "Loop Body")
                    # Edge from end of body back to loop condition (simplified)
                    if body_entry_node_id: # If body wasn't empty
                         # This is a simplification; finding the *actual* last node of body is harder
                         # For visualization, an edge from loop construct to its body start is okay
                         pass # Edges back are complex for this simple CFG

                prev_node_id = control_node_id # Next connection point after control structure
                entry_edge_label = ""
                idx += 1
        
        # Add any remaining statements in a final block
        if current_block_stmts:
            block_node_id = new_node_id()
            label = "\n".join(format_stmt_for_cfg(s) for s in current_block_stmts)
            dot.node(block_node_id, label=label, shape="box")
            if prev_node_id:
                dot.edge(prev_node_id, block_node_id, label=entry_edge_label)
            return block_node_id
        return prev_node_id


    # Determine the entry point for CFG building
    start_node_id = new_node_id()
    dot.node(start_node_id, label="START", shape="ellipse")

    if is_ssa: # List of SSA statements
        build_cfg_nodes(ast_or_ssa_stmts, start_node_id)
    else: # AST program node ("program", list_of_stmts)
        if ast_or_ssa_stmts and ast_or_ssa_stmts[0] == "program":
            build_cfg_nodes(ast_or_ssa_stmts[1], start_node_id)
        else: # Or direct list of statements if program wrapper is removed
            build_cfg_nodes(ast_or_ssa_stmts, start_node_id)

    # Add an END node (optional, good for completeness)
    # Find terminal nodes (nodes with no outgoing edges other than loopbacks)
    # This is simplified: connect last processed node to END if it's not part of a complex structure already handled
    # end_node_id = new_node_id()
    # dot.node(end_node_id, label="END", shape="ellipse")
    # if node_counter > 1 and prev_node_id != start_node_id : # check if anything was added beyond start
    #     dot.edge(prev_node_id, end_node_id) # Connect last processed node to END

    return dot


# --- 7. Streamlit GUI ---
st.set_page_config(layout="wide")
st.title("Program Analysis Tool (Formal Methods)")

# --- Input Area ---
st.sidebar.header("Program Input")
mode = st.sidebar.radio("Select Mode", ("Verification", "Equivalence"))
unroll_depth = st.sidebar.number_input("Loop Unroll Depth", min_value=0, value=3, step=1)

program_code_1 = st.sidebar.text_area("Program 1 Code", height=200, value="""
n := 3;
arr := [10, 20, 5];
for (i := 0; i < n; i := i + 1;) {
    min_idx := i;
    for (j := i + 1; j < n; j := j + 1;) {
        if (arr[j] < arr[min_idx]) {
            min_idx := j;
        }
    }
    // Swap
    temp := arr[i];
    arr[i] := arr[min_idx];
    arr[min_idx] := temp;
}
assert(arr[0] <= arr[1]);
assert(arr[1] <= arr[2]);
// assert(arr[0] < arr[1]); // This might fail for [5,5,10]
""") # Example Selection Sort

program_code_2_disabled = (mode == "Verification")
program_code_2 = st.sidebar.text_area("Program 2 Code (for Equivalence)", height=200, disabled=program_code_2_disabled, value="""
n := 3;
arr := [10, 20, 5]; // Same input for equivalence
x := 0;
while (x < n) {
    y := x + 1;
    while (y < n) {
        if (arr[y] < arr[x]) {
            // Swap arr[x] and arr[y]
            temp := arr[x];
            arr[x] := arr[y];
            arr[y] := temp;
        }
        y := y + 1;
    }
    x := x + 1;
}
// No assertions needed for equivalence, comparing final state of 'arr' and 'n'
""") # Example Bubble Sort (ish)

if st.sidebar.button("Analyze"):
    # --- Common Processing ---
    def process_program(code, unroll_d, program_name="Program"):
        st.header(f"Analysis for {program_name}")
        
        st.subheader("1. Original Code")
        st.code(code, language='plaintext')

        try:
            ast = parser.parse(code)
        except Exception as e:
            st.error(f"Parsing Error in {program_name}: {e}")
            return None
        
        st.subheader(f"2. Abstract Syntax Tree (AST) - {program_name}")
        st.json(ast) # Show raw AST

        st.subheader(f"3. Original Code CFG - {program_name}")
        try:
            original_cfg = generate_cfg(ast, is_ssa=False)
            st.graphviz_chart(original_cfg)
        except Exception as e:
            st.warning(f"Could not generate original CFG for {program_name}: {e}")


        st.subheader(f"4. Unrolled AST (Depth: {unroll_d}) - {program_name}")
        unrolled_ast = unroll_loops_in_ast(copy.deepcopy(ast), unroll_d)
        st.json(unrolled_ast) # Show unrolled AST (can be verbose)
        # TODO: Pretty print unrolled code from unrolled_ast

        st.subheader(f"5. SSA Form (after unrolling) - {program_name}")
        ssa_converter = SSATransformer()
        try:
            ssa_program = ssa_converter.transform(unrolled_ast)
            pretty_ssa = "\n".join(pretty_print_ssa_stmts(ssa_program))
            st.code(pretty_ssa, language='plaintext')
        except Exception as e:
            st.error(f"SSA Conversion Error in {program_name}: {e}")
            return None

        st.subheader(f"6. Unrolled SSA CFG - {program_name}")
        try:
            ssa_cfg = generate_cfg(ssa_program, is_ssa=True)
            st.graphviz_chart(ssa_cfg)
        except Exception as e:
            st.warning(f"Could not generate SSA CFG for {program_name}: {e}")


        st.subheader(f"7. SMT Constraints - {program_name}")
        smt_gen = SMTGenerator(unroll_depth_for_array_size_heuristic=unroll_d)
        try:
            solver, program_assertions, smt_code, z3_vars, z3_arrays = smt_gen.generate_smt(ssa_program)
            st.code(smt_code, language='z3')
            final_vars, final_arrays = smt_gen.get_final_ssa_vars(ssa_converter)
            return solver, program_assertions, z3_vars, z3_arrays, final_vars, final_arrays, smt_gen.max_array_elements
        except Exception as e:
            st.error(f"SMT Generation Error in {program_name}: {e}")
            # import traceback
            # st.text(traceback.format_exc())
            return None

    # --- Mode-Specific Logic ---
    results1 = process_program(program_code_1, unroll_depth, "Program 1")

    if mode == "Verification":
        if results1:
            solver, program_assertions, z3_vars, z3_arrays, _, _, _ = results1
            st.header("Verification Results (Program 1)")

            if not program_assertions:
                st.warning("No assertions found in Program 1 to verify.")
            else:
                all_assertions_hold = True
                failed_assertions_details = []

                for i, p_assert in enumerate(program_assertions):
                    solver.push()
                    solver.add(z3.Not(p_assert)) # Check if (NOT assertion) is satisfiable
                    
                    check_result = solver.check()
                    assertion_holds = (check_result == z3.unsat)
                    
                    if assertion_holds:
                        st.success(f"Assertion {i+1}: Holds (`{solver.to_smt2(p_assert)}`)")
                        
                        # Try to find one example where postcondition holds
                        solver.pop() # Remove (Not p_assert)
                        solver.push()
                        solver.add(p_assert) # Add original assertion
                        if solver.check() == z3.sat:
                            model = solver.model()
                            example_str = "Example where assertion holds:\n"
                            for var_name_ssa, z3_var_obj in z3_vars.items():
                                try: example_str += f"  {var_name_ssa} = {model.eval(z3_var_obj, model_completion=True)}\n"
                                except: pass # Var might not be in model if not constrained
                            for arr_name_ssa, z3_arr_obj in z3_arrays.items():
                                try: 
                                    # Try to print some elements
                                    arr_str = f"  {arr_name_ssa} = ["
                                    for k_idx in range(min(5, unroll_depth+2)): # Print a few elements
                                        try: arr_str += f"{model.eval(z3.Select(z3_arr_obj, k_idx), model_completion=True)}, "
                                        except: arr_str += "?, " # Element might not be constrained
                                    arr_str = arr_str.rstrip(", ") + "]"
                                    example_str += arr_str + "\n"
                                except: pass
                            st.text(example_str)
                        else:
                            st.info(f"Could not find a specific model where assertion {i+1} holds (though it's proven valid).")
                        solver.pop() # Clean up stack

                    else: # Assertion failed (check_result == z3.sat for Not(p_assert))
                        st.error(f"Assertion {i+1}: Fails (`{solver.to_smt2(p_assert)}`)")
                        all_assertions_hold = False
                        counterexamples = []
                        for _ in range(2): # Find up to 2 counterexamples
                            if solver.check() == z3.sat:
                                model = solver.model()
                                cex_str = "Counterexample:\n"
                                current_cex_constraints = []
                                for var_name_ssa, z3_var_obj in z3_vars.items():
                                    try: 
                                        val = model.eval(z3_var_obj, model_completion=True)
                                        cex_str += f"  {var_name_ssa} = {val}\n"
                                        current_cex_constraints.append(z3_var_obj != val) # For next distinct CEX
                                    except: pass
                                for arr_name_ssa, z3_arr_obj in z3_arrays.items():
                                    try:
                                        arr_s = f"  {arr_name_ssa} = ["
                                        arr_vals_for_next_cex = []
                                        for k_idx in range(min(5, unroll_depth+2)):
                                            try: 
                                                val = model.eval(z3.Select(z3_arr_obj, k_idx), model_completion=True)
                                                arr_s += f"{val}, "
                                                arr_vals_for_next_cex.append(z3.Select(z3_arr_obj, k_idx) != val)
                                            except: arr_s += "?, "
                                        arr_s = arr_s.rstrip(", ") + "]"
                                        cex_str += arr_s + "\n"
                                        if arr_vals_for_next_cex:
                                            current_cex_constraints.append(z3.Or(arr_vals_for_next_cex))
                                    except: pass
                                counterexamples.append(cex_str)
                                if current_cex_constraints: # Add constraint to find a new model
                                    solver.add(z3.Or(current_cex_constraints)) 
                                else: # No variables to make distinct, break
                                    break
                            else: # No more counterexamples
                                break
                        for cex in counterexamples: st.text(cex)
                        failed_assertions_details.append(f"Assertion {i+1} ({solver.to_smt2(p_assert)})")
                        solver.pop() # Clean up from Not(p_assert)
                
                if all_assertions_hold:
                    st.balloons()
                    st.success("All assertions in Program 1 hold!")
                else:
                    st.error("Some assertions failed: " + ", ".join(failed_assertions_details))


    elif mode == "Equivalence":
        results2 = process_program(program_code_2, unroll_depth, "Program 2")
        st.header("Equivalence Checking Results")

        if results1 and results2:
            solver1, _, z3_vars1, z3_arrays1, final_vars1, final_arrays1, max_el1 = results1
            solver2, _, z3_vars2, z3_arrays2, final_vars2, final_arrays2, max_el2 = results2
            
            # Combine solvers/constraints. Assume inputs are symbolic and the same initially.
            # For now, we assume variables with same original name are inputs if not assigned first.
            # A more robust way is to declare specific inputs symbolically.
            # Here, we take all constraints from both.
            
            s_eq = z3.Solver()
            s_eq.add(solver1.assertions()) # Constraints from prog1
            s_eq.add(solver2.assertions()) # Constraints from prog2

            # Identify common output variables (based on original names)
            common_orig_vars = set(final_vars1.keys()).intersection(set(final_vars2.keys()))
            common_orig_arrays = set(final_arrays1.keys()).intersection(set(final_arrays2.keys()))
            
            if not common_orig_vars and not common_orig_arrays:
                st.warning("No common variables/arrays found to compare for equivalence.")
            else:
                equivalence_conditions = []
                for orig_name in common_orig_vars:
                    ssa_name1 = final_vars1[orig_name]
                    ssa_name2 = final_vars2[orig_name]
                    # Get Z3 objects (must exist if they are final vars from SMT gen)
                    z3_v1 = z3_vars1.get(ssa_name1)
                    z3_v2 = z3_vars2.get(ssa_name2)
                    if z3_v1 is not None and z3_v2 is not None:
                        equivalence_conditions.append(z3_v1 == z3_v2)
                        st.write(f"Comparing final var: `{orig_name}` ({ssa_name1} vs {ssa_name2})")
                    else:
                        st.warning(f"Could not find Z3 var for {orig_name} (SSA: {ssa_name1} or {ssa_name2})")


                max_elements_to_compare = min(max_el1, max_el2, unroll_depth + 2, 5) # Heuristic
                for orig_name in common_orig_arrays:
                    ssa_arr_name1 = final_arrays1[orig_name]
                    ssa_arr_name2 = final_arrays2[orig_name]
                    z3_arr1 = z3_arrays1.get(ssa_arr_name1)
                    z3_arr2 = z3_arrays2.get(ssa_arr_name2)

                    if z3_arr1 is not None and z3_arr2 is not None:
                        st.write(f"Comparing final array: `{orig_name}` ({ssa_arr_name1} vs {ssa_arr_name2}) up to {max_elements_to_compare} elements.")
                        # Compare element-wise up to a certain bound (e.g., unroll_depth or a fixed small N)
                        # This is a form of bounded equivalence for arrays.
                        # For full array equivalence: equivalence_conditions.append(z3_arr1 == z3_arr2)
                        # However, element-wise can be more useful for counterexamples.
                        for i in range(max_elements_to_compare): 
                            equivalence_conditions.append(z3.Select(z3_arr1, i) == z3.Select(z3_arr2, i))
                    else:
                        st.warning(f"Could not find Z3 array for {orig_name} (SSA: {ssa_arr_name1} or {ssa_arr_name2})")
                
                if not equivalence_conditions:
                    st.error("Could not formulate any equivalence conditions.")
                else:
                    # Check if (P1_constraints AND P2_constraints AND NOT(outputs_equal)) is SAT
                    s_eq.add(z3.Not(z3.And(equivalence_conditions)))
                    
                    eq_check_result = s_eq.check()

                    if eq_check_result == z3.unsat:
                        st.success(f"Programs are equivalent for the checked variables/arrays (up to unroll depth {unroll_depth} and array comparison depth {max_elements_to_compare}).")
                        st.balloons()
                        
                        # Find one example of common behavior
                        s_eq.reset()
                        s_eq.add(solver1.assertions())
                        s_eq.add(solver2.assertions())
                        s_eq.add(z3.And(equivalence_conditions)) # Assert they ARE equal
                        if s_eq.check() == z3.sat:
                            model = s_eq.model()
                            example_str = "Example of equivalent behavior:\n"
                            # Display values for common final vars/arrays
                            for orig_name in common_orig_vars:
                                ssa_name1 = final_vars1[orig_name]
                                z3_v1 = z3_vars1.get(ssa_name1)
                                if z3_v1:
                                    try: example_str += f"  {orig_name} (from P1: {ssa_name1}) = {model.eval(z3_v1, model_completion=True)}\n"
                                    except: pass
                            for orig_name in common_orig_arrays:
                                ssa_arr_name1 = final_arrays1[orig_name]
                                z3_arr1 = z3_arrays1.get(ssa_arr_name1)
                                if z3_arr1:
                                    try:
                                        arr_s = f"  {orig_name} (from P1: {ssa_arr_name1}) = ["
                                        for k_idx in range(max_elements_to_compare):
                                            try: arr_s += f"{model.eval(z3.Select(z3_arr1, k_idx), model_completion=True)}, "
                                            except: arr_s += "?, "
                                        arr_s = arr_s.rstrip(", ") + "]"
                                        example_str += arr_s + "\n"
                                    except: pass
                            st.text(example_str)
                        else:
                            st.info("Could not find a specific model for equivalent behavior (though proven equivalent).")

                    else: # eq_check_result == z3.sat (Not(outputs_equal) is SAT)
                        st.error(f"Programs are NOT equivalent (up to unroll depth {unroll_depth} and array comparison depth {max_elements_to_compare}).")
                        counterexamples = []
                        for _ in range(2): # Find up to 2 counterexamples
                             if s_eq.check() == z3.sat:
                                model = s_eq.model()
                                cex_str = "Counterexample (inputs leading to different outputs):\n"
                                current_cex_constraints = [] # For finding distinct CEX

                                # Show initial symbolic inputs if any, or relevant vars
                                # This part is tricky: need to identify "inputs"
                                # For now, just show all final vars from both programs
                                cex_str += "  Program 1 final state:\n"
                                for orig_name, ssa_name1 in final_vars1.items():
                                    z3_v1 = z3_vars1.get(ssa_name1)
                                    if z3_v1:
                                        try: 
                                            val = model.eval(z3_v1, model_completion=True)
                                            cex_str += f"    {orig_name} ({ssa_name1}) = {val}\n"
                                            current_cex_constraints.append(z3_v1 != val)
                                        except: pass
                                for orig_name, ssa_arr_name1 in final_arrays1.items():
                                     z3_arr1 = z3_arrays1.get(ssa_arr_name1)
                                     if z3_arr1:
                                        try:
                                            arr_s = f"    {orig_name} ({ssa_arr_name1}) = ["
                                            arr_vals_for_next_cex = []
                                            for k_idx in range(max_elements_to_compare):
                                                try: 
                                                    val = model.eval(z3.Select(z3_arr1, k_idx), model_completion=True)
                                                    arr_s += f"{val}, "
                                                    arr_vals_for_next_cex.append(z3.Select(z3_arr1, k_idx) != val)
                                                except: arr_s += "?, "
                                            arr_s = arr_s.rstrip(", ") + "]"
                                            cex_str += arr_s + "\n"
                                            if arr_vals_for_next_cex: current_cex_constraints.append(z3.Or(arr_vals_for_next_cex))
                                        except: pass

                                cex_str += "  Program 2 final state:\n"
                                for orig_name, ssa_name2 in final_vars2.items():
                                    z3_v2 = z3_vars2.get(ssa_name2)
                                    if z3_v2:
                                        try: 
                                            val = model.eval(z3_v2, model_completion=True)
                                            cex_str += f"    {orig_name} ({ssa_name2}) = {val}\n"
                                            current_cex_constraints.append(z3_v2 != val) # Add to distinct CEX constraints
                                        except: pass
                                for orig_name, ssa_arr_name2 in final_arrays2.items():
                                     z3_arr2 = z3_arrays2.get(ssa_arr_name2)
                                     if z3_arr2:
                                        try:
                                            arr_s = f"    {orig_name} ({ssa_arr_name2}) = ["
                                            arr_vals_for_next_cex = []
                                            for k_idx in range(max_elements_to_compare):
                                                try:
                                                    val = model.eval(z3.Select(z3_arr2, k_idx), model_completion=True)
                                                    arr_s += f"{val}, "
                                                    arr_vals_for_next_cex.append(z3.Select(z3_arr2, k_idx) != val)
                                                except: arr_s += "?, "
                                            arr_s = arr_s.rstrip(", ") + "]"
                                            cex_str += arr_s + "\n"
                                            if arr_vals_for_next_cex: current_cex_constraints.append(z3.Or(arr_vals_for_next_cex))
                                        except: pass
                                
                                counterexamples.append(cex_str)
                                if current_cex_constraints:
                                    s_eq.add(z3.Or(current_cex_constraints)) # To find a new model
                                else: break # Cannot make model distinct
                             else: # No more counterexamples
                                break
                        for cex in counterexamples: st.text(cex)
        else:
            st.error("One or both programs failed to process. Cannot perform equivalence check.")