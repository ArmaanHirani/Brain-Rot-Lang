import re

# Tokenizer
TOKENS = {
    'ASSIGN': r'w_comms',  # Matches assignment
    'NUMBER': r'\-?\d+',   # Matches numbers, including negatives
    'STRING': r'"[^"]*"',  # Matches strings in double quotes
    'PRINT': r'yap',       # Matches print statements
    'ARITH_OP': r'aura|skibidi|sigma|beta|fanumTax',  # Matches arithmetic operations, including modulo
    'COMPARISON': r'vibes|ratio|based|kindaRatio|kindaBased',  # Matches equality and relational operators
    'COMMA': r',',         # Matches commas
    'IF': r'lowkey',       # Matches the keyword 'if'
    'ELSE': r'highkey',    # Matches the keyword 'else'
    'LOOP': r'loop',       # Matches 'loop' for for-loops
    'FROM': r'runItBack',       # Matches 'from' in for-loops
    'TO': r'to',           # Matches 'to' in for-loops
    'WHILE': r'while',  # Matches the keyword 'while'
    'IDENTIFIER': r'[a-zA-Z_]\w*',  # Matches variable names
    'NEWLINE': r'\n',      # Matches newline characters
    'SPACE': r'[ \t]+',    # Matches spaces and tabs
    'COMMENT': r'#.*',     # Matches comments
}


# Tokenizer Function
def tokenize(code):
    tokens = []
    pos = 0
    indent_stack = [0]  # Stack to track indentation levels

    while pos < len(code):
        match = None
        for token_type, pattern in TOKENS.items():
            regex = re.compile(pattern)
            match = regex.match(code, pos)
            if match:
                text = match.group(0)
                if token_type == 'COMMENT':  # Skip comments
                    pos = match.end()
                    break
                elif token_type == 'NEWLINE':
                    tokens.append((token_type, text.strip()))
                    pos = match.end()
                    # Handle indentation for the next line
                    next_line = re.match(r'[ \t]*', code[pos:])
                    if next_line:
                        indent_level = len(next_line.group(0))
                        current_indent = indent_stack[-1]
                        if indent_level > current_indent:
                            tokens.append(('INDENT', ''))
                            indent_stack.append(indent_level)
                        while indent_level < current_indent:
                            tokens.append(('DEDENT', ''))
                            indent_stack.pop()
                            current_indent = indent_stack[-1]
                    break
                else:
                    tokens.append((token_type, text.strip()))
                    pos = match.end()
                    break
        if not match:
            raise SyntaxError(f"Unexpected character: {code[pos]}")
    # Close any remaining indentation levels
    while len(indent_stack) > 1:
        tokens.append(('DEDENT', ''))
        indent_stack.pop()

    return [token for token in tokens if token[0] != 'SPACE']  # Remove spaces


# AST Node Classes
class ASTNode:
    pass

class NumberNode(ASTNode):
    def __init__(self, value):
        self.value = int(value)

class VarNode(ASTNode):
    def __init__(self, name):
        self.name = name

class AssignNode(ASTNode):
    def __init__(self, var_name, value):
        self.var_name = var_name
        self.value = value

class PrintNode(ASTNode):
    def __init__(self, values):
        self.values = values  # Store a list of expressions

class BinaryOpNode(ASTNode):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class StringNode(ASTNode):
    def __init__(self, value):
        self.value = value

class IfNode(ASTNode):
    def __init__(self, condition, if_body, else_body=None):
        self.condition = condition      # The condition to evaluate
        self.if_body = if_body          # The body of the if-branch
        self.else_body = else_body      # The body of the else-branch (optional)

class ForNode(ASTNode):
    def __init__(self, var_name, start, end, body):
        self.var_name = var_name
        self.start = start
        self.end = end
        self.body = body

class WhileNode(ASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body





# Parser
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current_token(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected_type=None):
        token = self.current_token()
        if not token:
            raise SyntaxError("Unexpected end of input")
        if expected_type and token[0] != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {token[0]}")
        self.pos += 1
        return token

    def parse(self):
        statements = []
        while self.current_token():
            if self.current_token()[0] == 'NEWLINE':
                self.consume('NEWLINE')
                continue
            statements.append(self.parse_statement())
        return statements

    def parse_statement(self):
        token = self.current_token()
        if token[0] == 'ASSIGN':
            return self.parse_assignment()
        elif token[0] == 'PRINT':
            return self.parse_print()
        elif token[0] == 'IF':
            return self.parse_if()
        elif token[0] == 'LOOP':
            return self.parse_loop()
        elif token[0] == 'WHILE':  # Parse 'while' statements
            return self.parse_while()
        else:
            raise SyntaxError(f"Unexpected token: {token}")



    def parse_assignment(self):
        self.consume('ASSIGN')  # Consume 'w_comms'
        var_name = self.consume('IDENTIFIER')[1]  # Get the variable name
        self.consume('TO')  # Consume 'to'
        value = self.parse_expression()  # Parse the value to assign
        return AssignNode(var_name, value)


    def parse_print(self):
        self.consume('PRINT')  # Consume 'yap'
        values = [self.parse_expression()]  # Parse the first expression
        while self.current_token() and self.current_token()[0] == 'COMMA':
            self.consume('COMMA')  # Consume the comma
            values.append(self.parse_expression())  # Parse the next expression
        return PrintNode(values)

    def parse_expression(self):
        left = self.parse_term()
        while self.current_token() and self.current_token()[0] in ('ARITH_OP', 'COMPARISON'):
            op = self.consume(self.current_token()[0])[1]  # Consume ARITH_OP or COMPARISON
            right = self.parse_term()
            left = BinaryOpNode(left, op, right)
        return left


    def parse_term(self):
        token = self.current_token()
        if token[0] == 'NUMBER':
            self.consume('NUMBER')
            return NumberNode(token[1])
        elif token[0] == 'IDENTIFIER':
            self.consume('IDENTIFIER')
            return VarNode(token[1])
        elif token[0] == 'STRING':  # Handle strings
            self.consume('STRING')
            return StringNode(token[1][1:-1])  # Remove the enclosing quotes
        else:
            raise SyntaxError(f"Unexpected token in term: {token}")
        
    def parse_if(self):
        self.consume('IF')  # Consume 'if'
        condition = self.parse_expression()  # Parse the condition
        self.consume('NEWLINE')  # Consume the newline after 'if'
        self.consume('INDENT')  # Consume the indentation for the if-block

        # Parse the body of the if-block
        if_body = []
        while self.current_token() and self.current_token()[0] != 'DEDENT':
            if self.current_token()[0] == 'NEWLINE':  # Skip blank lines
                self.consume('NEWLINE')
                continue
            if_body.append(self.parse_statement())
        self.consume('DEDENT')  # Consume the dedentation after the if-block

        # Check for and parse the else-block
        else_body = None
        if self.current_token() and self.current_token()[0] == 'ELSE':
            self.consume('ELSE')  # Consume 'else'
            self.consume('NEWLINE')  # Consume the newline after 'else'
            self.consume('INDENT')  # Consume the indentation for the else-block
            else_body = []
            while self.current_token() and self.current_token()[0] != 'DEDENT':
                if self.current_token()[0] == 'NEWLINE':  # Skip blank lines
                    self.consume('NEWLINE')
                    continue
                else_body.append(self.parse_statement())
            self.consume('DEDENT')  # Consume the dedentation after the else-block

        return IfNode(condition, if_body, else_body)

    def parse_loop(self):
        self.consume('LOOP')  # Consume 'loop'
        var_name = self.consume('IDENTIFIER')[1]  # Get the loop variable
        self.consume('FROM')  # Consume 'from'
        start = self.parse_expression()  # Parse the start expression
        self.consume('TO')  # Consume 'to'
        end = self.parse_expression()  # Parse the end expression
        self.consume('NEWLINE')  # Consume the newline after the loop header
        self.consume('INDENT')  # Consume the indentation for the loop body

        # Parse the loop body
        body = []
        while self.current_token() and self.current_token()[0] != 'DEDENT':
            if self.current_token()[0] == 'NEWLINE':
                self.consume('NEWLINE')
                continue
            body.append(self.parse_statement())
        self.consume('DEDENT')  # Consume the dedentation after the loop body

        return ForNode(var_name, start, end, body)
    
    def parse_while(self):
        self.consume('WHILE')  # Consume 'while'
        condition = self.parse_expression()  # Parse the loop condition
        self.consume('NEWLINE')  # Consume the newline after 'while'
        self.consume('INDENT')  # Consume the indentation for the loop body

        # Parse the loop body
        body = []
        while self.current_token() and self.current_token()[0] != 'DEDENT':
            if self.current_token()[0] == 'NEWLINE':
                self.consume('NEWLINE')
                continue
            body.append(self.parse_statement())
        self.consume('DEDENT')  # Consume the dedentation after the loop body

        return WhileNode(condition, body)




# Interpreter
class Interpreter:
    def __init__(self):
        self.variables = {}

    def eval(self, node):
        if isinstance(node, NumberNode):
            return node.value
        elif isinstance(node, VarNode):
            if node.name not in self.variables:
                raise NameError(f"Variable '{node.name}' not defined")
            return self.variables[node.name]
        elif isinstance(node, AssignNode):
            value = self.eval(node.value)
            self.variables[node.var_name] = value
        elif isinstance(node, PrintNode):
            values = [self.eval(value) for value in node.values]
            print(" ".join(map(str, values)))
        elif isinstance(node, BinaryOpNode):
            left = self.eval(node.left)
            right = self.eval(node.right)
            if node.op == 'aura':  # Addition
                return left + right
            elif node.op == 'skibidi':  # Subtraction
                return left - right
            elif node.op == 'sigma':  # Multiplication
                return left * right
            elif node.op == 'beta':  # Division
                if right == 0:
                    raise ZeroDivisionError("Division by zero")
                return left // right
            elif node.op == 'fanumTax':  # Modulo
                if right == 0:
                    raise ZeroDivisionError("Modulo by zero")
                return left % right
            elif node.op == 'vibes':  # Equality
                return left == right
            elif node.op == 'ratio':  # Less than
                return left < right
            elif node.op == 'based':  # Greater than
                return left > right
            elif node.op == 'kindaRatio':  # Less than or equal to
                return left <= right
            elif node.op == 'kindaBased':  # Greater than or equal to
                return left >= right
            else:
                raise SyntaxError(f"Unknown operator: {node.op}")

        elif isinstance(node, StringNode):
            return node.value
        elif isinstance(node, IfNode):
            condition_result = self.eval(node.condition)
            if condition_result:
                for statement in node.if_body:
                    self.eval(statement)
            elif node.else_body:
                for statement in node.else_body:
                    self.eval(statement)
        elif isinstance(node, ForNode):
            start = self.eval(node.start)
            end = self.eval(node.end)
            if not isinstance(start, int) or not isinstance(end, int):
                raise TypeError("Loop bounds must be integers")
            for i in range(start, end + 1):
                self.variables[node.var_name] = i
                for statement in node.body:
                    self.eval(statement)
        elif isinstance(node, WhileNode):  # Add this block for while loops
            while self.eval(node.condition):
                for statement in node.body:
                    self.eval(statement)
        else:
            raise SyntaxError("Invalid syntax tree")

    def execute(self, tree):
        for node in tree:
            self.eval(node)


# Main function
def main():
    with open(r'C:\Users\User\Desktop\GenZLang\test_program.genz', 'r') as f:
        code = f.read()
    tokens = tokenize(code)
    parser = Parser(tokens)
    tree = parser.parse()
    interpreter = Interpreter()
    interpreter.execute(tree)

if __name__ == '__main__':
    main()
