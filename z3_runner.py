from z3 import Solver, parse_smt2_string, sat, Or, And

# Function to normalize code (if needed for your real use case)
def normalize_whitespace(code: str) -> str:
    return code.replace('\u00A0', ' ').replace('\u200B', ' ')

# Function to generate SMT for a given program (mockup, replace with actual logic)
def generate_smt_for_program(code: str, unroll: int):
    # Replace this with actual logic to convert code to SSA, and then to SMT.
    # For the sake of simplicity, we will use mock SMT code.
    
    # Example SMT representation for a mock program
    smt = f"(declare-fun arr_0 () Int)\n"
    smt += f"(assert (= arr_0 {unroll}))\n"
    return smt

# Function to run Z3 and get models (satisfying assignments)
def run_z3_models(smt1: str, smt2: str, max_models: int = 2):
    solver = Solver()
    solver.add(parse_smt2_string(smt1))
    solver.add(parse_smt2_string(smt2))
    
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

# Function to compare two programs
def compare_programs(code1: str, code2: str, unroll: int):
    smt1 = generate_smt_for_program(code1, unroll)
    smt2 = generate_smt_for_program(code2, unroll)
    
    # Compare programs using Z3 solver
    status, models = run_z3_models(smt1, smt2)
    
    if status == 'UNSAT':
        print("✅ Programs are equivalent. Here's an example input where they match:")
        # Witness: an example where outputs match
        solver = Solver()
        solver.add(parse_smt2_string(smt1))
        solver.add(parse_smt2_string(smt2))
        if solver.check() == sat:
            m = solver.model()
            for d in m.decls():
                print(f"{str(d)} = {m[d]}")
    else:
        print("❌ Programs are not equivalent. Here are counterexamples:")
        for m in models:
            for d in m.decls():
                print(f"{str(d)} = {m[d]}")

# Demo usage
if __name__ == "__main__":
    code1 = """
    n := 3;
    x := 4;
    assert(n > x);
    """
    code2 = """
    n := 3;
    x := 4;
    assert(n > x);
    """
    unroll = 3  # Example unroll value

    print("Comparing Program 1 and Program 2:")
    compare_programs(code1, code2, unroll)
