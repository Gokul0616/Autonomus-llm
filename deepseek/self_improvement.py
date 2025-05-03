# self_improvement.py
import ast
import docker
import time
import logging
import resource
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import re
import torch
from reward_model import RewardModel
import deepmodel

logger = logging.getLogger(__name__)

class CodeValidator:
    def __init__(self, timeout=5, memory_limit=512, use_docker=True):
        self.timeout = timeout
        self.memory_limit = memory_limit * 1024 * 1024  # Convert MB to bytes
        self.use_docker = use_docker
        self.docker_client = docker.from_env() if use_docker else None
        
        # Security patterns
        self.forbidden_patterns = [
            r"os\.system", r"subprocess", r"open\(|write\(", 
            r"import\s+(os|sys|subprocess)", r"__import__", r"eval\("
        ]

    def _safe_ast_parse(self, code: str) -> bool:
        """Static analysis for forbidden operations"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        if alias.name in ['os', 'sys', 'subprocess']:
                            return False
                if isinstance(node, ast.Call):
                    if any(getattr(node.func, 'id', None) == fn for fn in ['eval', 'exec']):
                        return False
            return True
        except:
            return False

    def _validate_syntax(self, code: str) -> bool:
        """Deep syntax validation"""
        if any(re.search(patt, code) for patt in self.forbidden_patterns):
            return False
        return self._safe_ast_parse(code)

    def _run_in_container(self, code: str) -> Tuple[bool, str, float]:
        """Docker-based execution with resource limits"""
        container = None
        try:
            container = self.docker_client.containers.run(
                "python:3.9-slim",
                command=f"python -c 'import resource; resource.setrlimit(resource.RLIMIT_AS, ({self.memory_limit}, {self.memory_limit})); exec(open('code.py').read())'",
                volumes={Path(tempfile.mkdtemp()).absolute(): {'bind': '/app', 'mode': 'ro'}},
                working_dir="/app",
                mem_limit=f"{self.memory_limit}b",
                network_mode="none",
                detach=True
            )
            
            start_time = time.time()
            result = container.wait(timeout=self.timeout)
            elapsed = time.time() - start_time
            
            logs = container.logs().decode()
            success = result['StatusCode'] == 0
            return success, logs, elapsed
            
        except docker.errors.ContainerError as e:
            return False, str(e), 0
        finally:
            if container:
                container.remove(force=True)

    def _run_in_subprocess(self, code: str) -> Tuple[bool, str, float]:
        """Secure subprocess execution with resource limits"""
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = Path(tmpdir) / "code.py"
            with open(code_path, "w") as f:
                f.write(code)
            
            try:
                start_time = time.time()
                process = subprocess.run(
                    ["python", str(code_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=self.timeout,
                    preexec_fn=lambda: resource.setrlimit(
                        resource.RLIMIT_AS, 
                        (self.memory_limit, self.memory_limit)
                    )
                )
                elapsed = time.time() - start_time
                
                success = process.returncode == 0
                return success, process.stdout.decode(), elapsed
            except subprocess.TimeoutExpired:
                return False, "Timeout exceeded", 0

    def validate_code(self, code: str, tests: str = None) -> Dict:
        """Production-grade code validation with security checks"""
        validation_result = {
            "valid_syntax": False,
            "execution_success": False,
            "test_passed": False,
            "execution_time": 0.0,
            "output": "",
            "error": ""
        }

        try:
            # Phase 1: Static analysis
            if not self._validate_syntax(code):
                raise SecurityError("Forbidden operations detected")
            
            # Phase 2: Execution validation
            if self.use_docker and self.docker_client:
                exec_success, output, elapsed = self._run_in_container(code)
            else:
                exec_success, output, elapsed = self._run_in_subprocess(code)
            
            validation_result.update({
                "valid_syntax": True,
                "execution_success": exec_success,
                "execution_time": elapsed,
                "output": output
            })

            # Phase 3: Test validation
            if tests and exec_success:
                test_result = self._run_tests(code, tests)
                validation_result["test_passed"] = test_result
                
            return validation_result
            
        except Exception as e:
            validation_result["error"] = str(e)
            return validation_result

class SelfImprovementEngine:
    def __init__(self, model, validator: CodeValidator):
        self.model = model
        self.validator = validator
        self.reward_buffer = []
        self.improvement_history = []
        self.reward_model = RewardModel(model).cuda()
        self.reward_model.load_state_dict(torch.load("reward_model.pth"))
        # RL parameters
        self.gamma = 0.99
        self.entropy_coef = 0.01
        self.llm= deepmodel
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # Add to SelfImprovementEngine class
    def learn_from_experience(self, experience):
        """Continuous learning from real-world interactions"""
        # Convert experience to training data
        dataset = self._create_dataset(experience)
        
        # Fine-tune model
        self.train_model(dataset)
        
        # Validate improvements
        if self.validate_improvements():
            self.versioner.save(self.model)
            
    def _create_dataset(self, experience):
        return {
            "inputs": experience["state"],
            "targets": experience["action"],
            "rewards": experience["reward"]
        }
    def add_new_tool(self, tool_name: str, spec: str):
        # 1) prompt LLM to generate Python code for the new Tool subclass
        code = self.llm.generate_text(f"Write a HardwareTool subclass called {tool_name} that {spec}")
        # 2) validate syntax and safety
        result = self.validator.run(code)
        if result["success"]:
            # 3) exec the code in a safe namespace
            namespace = {}
            exec(code, namespace)
            new_tool_cls = namespace[tool_name]
            # 4) instantiate and register
            self.agent.tools[tool_name] = new_tool_cls(**self._infer_constructor_args(new_tool_cls))
            return True
        return False
    def generate_improvement(self, task_description: str):
        """Full self-improvement loop"""
        # Generate code and tests
        generated_code = self._generate_code(task_description)
        generated_tests = self._generate_tests(task_description, generated_code)
        
        # Validate implementation
        validation = self.validator.validate_code(generated_code, generated_tests)
        
        # Calculate reward
        reward = self._calculate_reward(validation)
        self.reward_buffer.append(reward)
        
        # Update model
        self._reinforcement_update(reward)
        
        # Log improvement
        self.improvement_history.append({
            "task": task_description,
            "code": generated_code,
            "validation": validation,
            "reward": reward
        })
        
        return validation

    def _calculate_reward(self, validation: Dict) -> float:
        return self.reward_model.compute_reward(validation)

    def _reinforcement_update(self, reward: float):
        """PPO-style policy update"""
        # Convert latest experience to policy gradient
        log_probs = torch.stack(self.model.last_log_probs)
        rewards = torch.tensor([reward], device=self.model.device)
        
        # Calculate advantage
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        
        # Normalize advantages
        advantages = returns - returns.mean()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate losses
        policy_loss = (-log_probs * advantages).mean()
        entropy_loss = -self.entropy_coef * (torch.exp(log_probs) * log_probs).mean()
        
        # Update model
        self.optimizer.zero_grad()
        (policy_loss + entropy_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

class SecurityError(Exception):
    pass