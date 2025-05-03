# autonomous/safety/security.py
class SecurityError(Exception):
    pass

class CodeSandbox:
    def __init__(self):
        self.allowed_actions = ["math", "statistics", "data_processing"]
        
    def execute_untrusted(self, code: str):
        """Safely execute untrusted code"""
        try:
            # Validate code
            self._validate_syntax(code)
            
            # Execute in restricted environment
            restricted_globals = {"__builtins__": None}
            return eval(code, restricted_globals)
            
        except Exception as e:
            raise SecurityError(f"Code execution failed: {str(e)}")

    def _validate_syntax(self, code):
        forbidden_keywords = ["import", "os", "sys", "subprocess"]
        for keyword in forbidden_keywords:
            if keyword in code:
                raise SecurityError(f"Forbidden keyword '{keyword}' detected")