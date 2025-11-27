import ast
import math
import numpy as np
from collections import Counter


def calculate_unique_variable_count(code_string: str) -> int:
    """
    Counts the number of unique variables in a code string.
    
    This function identifies variables from:
    - Simple assignments: x = 5
    - Tuple/list unpacking: a, b = 1, 2
    - Augmented assignments: x += 1
    - Function parameters: def func(x, y):
    - Loop variables: for i in range(10):
    - With statement variables: with open('file') as f:
    - Exception variables: except Exception as e:
    - Starred unpacking: *args, x = values
    
    Args:
        code_string: A string containing Python code
        
    Returns:
        The number of unique variables found in the code, or -1 if there's a syntax error
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        return np.nan  # Indicate syntax error
    
    variables = set()
    
    def extract_names(target):
        """Recursively extract variable names from assignment targets."""
        if isinstance(target, ast.Name):
            variables.add(target.id)
        elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
            for elt in target.elts:
                extract_names(elt)
        elif isinstance(target, ast.Starred):
            extract_names(target.value)
    
    for node in ast.walk(tree):
        # Regular assignments: x = 5, a, b = 1, 2
        if isinstance(node, ast.Assign):
            for target in node.targets:
                extract_names(target)
        
        # Augmented assignments: x += 1, a *= 2
        elif isinstance(node, ast.AugAssign):
            extract_names(node.target)
        
        # Function parameters: def func(x, y, z=5):
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args:
                variables.add(arg.arg)
            # *args and **kwargs
            if node.args.vararg:
                variables.add(node.args.vararg.arg)
            if node.args.kwarg:
                variables.add(node.args.kwarg.arg)
        
        # For loop variables: for i in range(10):
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            extract_names(node.target)
        
        # With statement variables: with open('file') as f:
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars:
                    extract_names(item.optional_vars)
        
        # Exception variables: except Exception as e:
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                variables.add(node.name)
        
        # Comprehension variables: [x for x in range(10)]
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for generator in node.generators:
                extract_names(generator.target)
    
    return len(variables)


def calculate_cyclomatic_complexity(code_string):
    """
    Calculates the Cyclomatic Complexity (M).
    This directly measures the Spartan principle of minimizing control structures.
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        return np.nan # Indicate syntax error

    # Start with 1 (the single entry/exit path)
    complexity = 1

    # Decision points increase complexity by 1
    # Note: ast.For, ast.While, ast.If, ast.ExceptHandler are main decision points
    # ast.BoolOp (and, or) also increases complexity
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler, ast.With)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            # 'and'/'or' operators contribute (k-1) where k is the number of values.
            # E.g., a and b and c has 2 decision points.
            complexity += len(node.values) - 1
        
        # We don't count function definitions as decision points here, 
        # as we are measuring complexity *within* the code block.

    return complexity


def calculate_halstead_volume(code_string):
    """
    Calculates the Halstead Volume (V) for a given Python code string
    by counting operators and operands based on AST analysis.
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        print("Error: Invalid Python syntax.")
        return np.nan

    # 1. Initialize Counters
    # We use Counters to track total occurrences (N1, N2)
    # The set of keys in the Counter will represent unique tokens (n1, n2)
    operator_counts = Counter()
    operand_counts = Counter()

    # Define common AST node types that represent operators (structural elements/actions)
    OPERATOR_NODES = (
        ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign, ast.AugAssign,
        ast.Return, ast.Yield, ast.Import, ast.ImportFrom, ast.Global,
        ast.Nonlocal, ast.Delete, ast.If, ast.For, ast.While, ast.With,
        ast.Raise, ast.Try, ast.Assert, ast.Break, ast.Continue
    )

    # Define nodes that represent explicit mathematical/logical operators
    EXPLICIT_OP_NODES = (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp)

    for node in ast.walk(tree):
        # A. Identify Operators (n1, N1)
        if isinstance(node, OPERATOR_NODES):
            # Use the node type name as the unique operator identifier
            operator_counts[type(node).__name__] += 1

        elif isinstance(node, EXPLICIT_OP_NODES):
            # For operators like +, -, *, /, ==, and, or, not
            # We count the operation type (e.g., Add, Sub, Eq, Not)
            if hasattr(node, 'op'):
                operator_counts[type(node.op).__name__] += 1
            # Also count the overall operation node
            operator_counts[type(node).__name__] += 1
        
        # B. Identify Operands (n2, N2)
        elif isinstance(node, ast.Name):
            # Variables used (Load context) or defined (Store context)
            operand_counts[node.id] += 1
        
        elif isinstance(node, ast.Constant):
            # Literals (numbers, strings, booleans, None)
            operand_counts[repr(node.value)] += 1
        
        # Note: This approach still simplifies tokens like '(', ',', etc., 
        # but provides a much more accurate structural count than the previous version.

    # 2. Calculate Halstead Components
    
    # n1: Unique Operators
    n1 = len(operator_counts)
    # N1: Total Operators
    N1 = sum(operator_counts.values())

    # n2: Unique Operands
    n2 = len(operand_counts)
    # N2: Total Operands
    N2 = sum(operand_counts.values())

    # Program Vocabulary (n) and Length (N)
    n = n1 + n2
    N = N1 + N2

    if n == 0:
        return 0.0

    # 3. Halstead Volume Calculation (V)
    # V = N * log2(n)
    V = N * math.log2(n)
    
    return V


def calculate_source_lines_of_code(code_string: str) -> int:
    """
    Calculates Source Lines of Code (SLOC).
    This measures the Spartan principle of minimizing vertical complexity (code length).

    It counts lines that are not blank and are not just comments.
    """
    lines = code_string.split('\n')
    sloc_count = 0
    
    # Simple approach to filter out blank and pure comment lines
    for line in lines:
        stripped_line = line.strip()
        
        # Check for blank line or pure comment line
        if not stripped_line or stripped_line.startswith('#'):
            continue
        
        # If the line contains anything else (code, comment after code), count it.
        sloc_count += 1
        
    return sloc_count

