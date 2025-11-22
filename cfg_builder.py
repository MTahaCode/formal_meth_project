# cfg_builder.py
from lark import Tree

class CFGNode:
    def __init__(self, id_val, label_prefix="B"):
        self.id = f"{label_prefix}{id_val}"
        self.statements_repr = [] 
        self.label_override = None

    def add_statement_repr(self, stmt_repr):
        self.statements_repr.append(stmt_repr)

    def get_label(self):
        if self.label_override:
            return self.label_override
        if not self.statements_repr:
            return self.id 
        content = "\\n".join(s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n") for s in self.statements_repr)
        return f"{self.id}\\n{content}"

class CFGBuilder:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.next_node_id_val = 0

    def _new_node(self, label_prefix="B"):
        node = CFGNode(self.next_node_id_val, label_prefix)
        self.next_node_id_val += 1
        self.nodes[node.id] = node
        return node

    def _add_edge(self, from_node_id, to_node_id, label=""):
        # Avoid duplicate edges if they might be added by different logic paths
        if not any(e[0] == from_node_id and e[1] == to_node_id and e[2] == label for e in self.edges):
            self.edges.append((from_node_id, to_node_id, label))

    def _ast_node_to_string(self, ast_node):
        if isinstance(ast_node, tuple):
            tag = ast_node[0]
            if tag == "assign":
                var_str = self._ast_node_to_string(ast_node[1])
                expr_str = self._ast_node_to_string(ast_node[2])
                return f"{var_str} := {expr_str}"
            elif tag == "var":
                return ast_node[1]
            elif tag == "num":
                return str(ast_node[1])
            elif tag == "arr_access":
                arr_str = self._ast_node_to_string(ast_node[1])
                idx_str = self._ast_node_to_string(ast_node[2])
                return f"{arr_str}[{idx_str}]"
            elif tag == "array_literal":
                items = ", ".join(self._ast_node_to_string(item) for item in ast_node[1:])
                return f"[{items}]"
            elif tag == "print":
                return f"print({self._ast_node_to_string(ast_node[1])})"
            elif tag == "assert":
                return f"assert({self._ast_node_to_string(ast_node[1])})"
            elif tag in ["arith", "cmp", "term", "or", "and"]:
                left_str = self._ast_node_to_string(ast_node[1])
                if len(ast_node) > 2 and ast_node[2]: 
                    if isinstance(ast_node[2], list) and ast_node[2] and isinstance(ast_node[2][0], tuple) and len(ast_node[2][0]) == 2: # For arith, cmp, term like ('op', val_expr)
                        op_str = ast_node[2][0][0]
                        right_str = self._ast_node_to_string(ast_node[2][0][1])
                        suffix = " ..." if len(ast_node[2]) > 1 else ""
                        return f"({left_str} {op_str} {right_str}{suffix})"
                    elif isinstance(ast_node[2], tuple): # For or, and like (op, right_expr)
                        op_str = "||" if tag == "or" else "&&" # Inferred op
                        right_str = self._ast_node_to_string(ast_node[2]) # ast_node[2] is the first *rest item
                        suffix = " ..." if len(ast_node) > 3 else ""
                        return f"({left_str} {op_str} {right_str}{suffix})"

                return left_str
            elif tag == "not_expr":
                return f"!({self._ast_node_to_string(ast_node[1])})"
            else:
                return str(tag) 
        elif isinstance(ast_node, str):
            return ast_node
        elif isinstance(ast_node, int):
            return str(ast_node)
        return "..."

    def build(self, list_of_stmts):
        self.nodes.clear()
        self.edges.clear()
        self.next_node_id_val = 0

        start_node = self._new_node("Entry")
        start_node.label_override = "Start"
        
        end_node = self._new_node("Exit") 
        end_node.label_override = "End"

        if not list_of_stmts:
            self._add_edge(start_node.id, end_node.id)
        else:
            self._process_stmt_list(list_of_stmts, start_node, end_node)
        
        return self.to_dot()

    def _process_stmt_list(self, stmts, current_node, exit_node_for_sequence):
        # `current_node` is the CFG node where this sequence of statements logically begins or continues.
        # `exit_node_for_sequence` is the CFG node to go to after this sequence naturally finishes.
        
        active_node = current_node

        for stmt_ast in stmts:
            stmt_type = stmt_ast[0]

            if stmt_type in ("assign", "print", "assert"):
                stmt_str = self._ast_node_to_string(stmt_ast)
                active_node.add_statement_repr(stmt_str)
            
            elif stmt_type == "if":
                cond_expr_str = self._ast_node_to_string(stmt_ast[1])
                active_node.add_statement_repr(f"IF ({cond_expr_str})")

                then_entry_node = self._new_node("If_Then")
                self._add_edge(active_node.id, then_entry_node.id, "True")
                
                merge_node = self._new_node("If_Merge")
                self._process_stmt_list(stmt_ast[2], then_entry_node, merge_node) # then_block

                if stmt_ast[3]: # Else block exists
                    else_entry_node = self._new_node("If_Else")
                    self._add_edge(active_node.id, else_entry_node.id, "False")
                    self._process_stmt_list(stmt_ast[3], else_entry_node, merge_node) # else_block
                else: 
                    self._add_edge(active_node.id, merge_node.id, "False")
                
                active_node = merge_node 

            elif stmt_type == "while":
                cond_node = self._new_node("W_Cond")
                if active_node.statements_repr or active_node.label_override == "Start": # If active_node has content, link it.
                    self._add_edge(active_node.id, cond_node.id)
                else: # active_node is an empty intermediate node, effectively becomes cond_node
                    # This case can be complex; safer to always create cond_node and link.
                    # If active_node was an empty merge point, it means previous flow leads to this while loop.
                    self._add_edge(active_node.id, cond_node.id)


                active_node = cond_node # Condition is now the active point
                cond_expr_str = self._ast_node_to_string(stmt_ast[1])
                active_node.add_statement_repr(f"WHILE ({cond_expr_str})")
                
                loop_body_entry_node = self._new_node("W_Body")
                self._add_edge(active_node.id, loop_body_entry_node.id, "True")

                after_loop_node = self._new_node("W_After") 
                self._add_edge(active_node.id, after_loop_node.id, "False (exit)")

                self._process_stmt_list(stmt_ast[2], loop_body_entry_node, active_node) # Loop body connects back to cond_node (active_node)
                
                active_node = after_loop_node

            elif stmt_type == "for":
                init_node = self._new_node("For_Init")
                if active_node.statements_repr or active_node.label_override == "Start":
                     self._add_edge(active_node.id, init_node.id)
                else: # active_node is an empty intermediate node
                    self._add_edge(active_node.id, init_node.id)


                init_stmt_str = self._ast_node_to_string(stmt_ast[1])
                init_node.add_statement_repr(init_stmt_str)

                cond_node = self._new_node("For_Cond")
                self._add_edge(init_node.id, cond_node.id)
                cond_expr_str = self._ast_node_to_string(stmt_ast[2])
                cond_node.add_statement_repr(f"FOR_COND ({cond_expr_str})")

                body_entry_node = self._new_node("For_Body")
                self._add_edge(cond_node.id, body_entry_node.id, "True")

                after_loop_node = self._new_node("For_After")
                self._add_edge(cond_node.id, after_loop_node.id, "False (exit)")

                update_node = self._new_node("For_Update")
                update_stmt_str = self._ast_node_to_string(stmt_ast[3])
                update_node.add_statement_repr(update_stmt_str)
                self._add_edge(update_node.id, cond_node.id) 

                self._process_stmt_list(stmt_ast[4], body_entry_node, update_node) 
                
                active_node = after_loop_node
            else:
                active_node.add_statement_repr(f"unknown_stmt: {stmt_type}")
        
        # Connect the final active_node of this list to the designated exit_node_for_sequence (fall-through)
        # unless active_node is already the exit_node or an edge already exists (e.g. from an empty else branch)
        if active_node.id != exit_node_for_sequence.id:
            self._add_edge(active_node.id, exit_node_for_sequence.id)

    def to_dot(self):
        dot_lines = ["digraph CFG {", 
                     "  rankdir=TB;", 
                     "  node [shape=box, fontname=Helvetica, fontsize=10, style=filled, fillcolor=lightyellow];", 
                     "  edge [fontname=Helvetica, fontsize=9];"]
        
        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            label = node.get_label()
            dot_lines.append(f'  "{node_id}" [label="{label}"];')

        for from_node_id, to_node_id, edge_label in self.edges:
            if edge_label:
                dot_lines.append(f'  "{from_node_id}" -> "{to_node_id}" [label="{edge_label}"];')
            else:
                dot_lines.append(f'  "{from_node_id}" -> "{to_node_id}";')
        dot_lines.append("}")
        return "\n".join(dot_lines)