# autonomous/safety/approval.py
from autonomous.safety.security import SecurityError
import re
from datetime import datetime, timedelta
class HumanApproval:
    def __init__(self):
        self.pending_requests = {}
        self.approved_tokens = {}
        
    def request_approval(self, action_description, justification=""):
        """Request human approval for an action and return approval token"""
        token = self.generate_token()
        self.pending_requests[token] = {
            "action": action_description,
            "timestamp": datetime.now(),
            "justification": justification,
            "approved": False
        }
        # In real implementation, this would trigger UI/notification
        print(f"Approval required for action: {action_description}")
        return token
    
    def approve_action(self, token):
        """Mark an action as approved by human"""
        if token in self.pending_requests:
            self.pending_requests[token]["approved"] = True
            self.approved_tokens[token] = datetime.now()
            
    def validate_token(self, token):
        """Check if token is valid and not expired"""
        if token in self.approved_tokens:
            # Tokens expire after 1 hour
            if datetime.now() - self.approved_tokens[token] < timedelta(hours=1):
                return True
            del self.approved_tokens[token]
        return False
    
    def generate_token(self):
        """Generate unique approval token"""
        import secrets
        return f"APPROVAL-{datetime.now().strftime('%Y%m%d')}-{secrets.token_hex(4)}".upper()

class ConstitutionalAI:
    def __init__(self):
        self.constraints = [
            "No system modification without approval",
            "Maintain user privacy",
            "Prevent physical harm"
        ]
        
        # Configuration for safety checks
        self.safety_parameters = {
            "max_movement_speed": 1.0,  # m/s
            "allowed_temperature_range": (0, 45),  # Celsius
            "restricted_directories": ["/system", "/bin", "/root"]
        }

    def validate(self, action):
        return all(
            self._check_constraint(constraint, action)
            for constraint in self.constraints
        )

    def _check_constraint(self, constraint, action):
        match constraint:
            case "No system modification without approval":
                return self._check_system_modification(action)
            
            case "Maintain user privacy":
                return self._check_privacy_violation(action)
            
            case "Prevent physical harm":
                return self._check_physical_safety(action)
            
            case _:
                # Default deny for unknown constraints
                raise SecurityError(f"Unknown constraint: {constraint}")

    def _check_system_modification(self, action):
        # Check for system-altering actions
        system_keywords = ["delete", "modify", "update", "install", "uninstall"]
        if any(kw in action.get("description", "").lower() for kw in system_keywords):
            # Verify human approval exists
            if not action.get("approval_token"):
                raise SecurityError("System modification requires approval token")
            
            # Validate token format (example implementation)
            if not re.match(r"APPROVAL-\d{4}-[A-Z]{8}", action["approval_token"]):
                raise SecurityError("Invalid approval token format")
        
        # Check file path restrictions
        if "file_path" in action:
            return not any(
                action["file_path"].startswith(restricted)
                for restricted in self.safety_parameters["restricted_directories"]
            )
        
        return True

    def _check_privacy_violation(self, action):
        # Check for PII patterns
        pii_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b(?:\+?1[-. ]?)?\(?[2-9][0-9]{2}\)?[-. ]?[2-9][0-9]{2}[-. ]?[0-9]{4}\b"  # Phone
        ]
        
        # Check action data
        for field in ["parameters", "description", "code"]:
            if field in action:
                content = str(action[field])
                if any(re.search(pattern, content) for pattern in pii_patterns):
                    raise SecurityError(f"Potential PII detected in {field}")
        
        return True

    def _check_physical_safety(self, action):
        # Check hardware parameters
        if "hardware_command" in action:
            cmd = action["hardware_command"]
            
            # Movement safety
            if "speed" in cmd:
                if cmd["speed"] > self.safety_parameters["max_movement_speed"]:
                    raise SecurityError(f"Speed exceeds safe limit: {cmd['speed']} > {self.safety_parameters['max_movement_speed']}")
            
            # Temperature safety
            if "temperature" in cmd:
                min_temp, max_temp = self.safety_parameters["allowed_temperature_range"]
                if not (min_temp <= cmd["temperature"] <= max_temp):
                    raise SecurityError(f"Temperature out of safe range: {cmd['temperature']}")
            
            # Collision prevention
            if "movement" in cmd and not cmd.get("collision_check"):
                raise SecurityError("Movement commands require collision check")
        
        return True