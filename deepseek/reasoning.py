# reasoning.py
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from enum import Enum
import json
from dataclasses import dataclass, field
from typing import Literal

from deepmodel import GPTModel  # Import your existing model

logger = logging.getLogger(__name__)

class ThoughtType(Enum):
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    ACTION = "action"
    VALIDATION = "validation"

class ActionType(Enum):
    CODE_GENERATION = "code_generation"
    CODE_VALIDATION = "code_validation"
    RESEARCH = "web_search"
    MEMORY_UPDATE = "memory_update"

# @dataclass
# class PlanStep:
#     thought: str
#     type: ThoughtType
#     state: Dict[str, Any]
#     parent: Optional['PlanStep'] = None
#     children: List['PlanStep'] = None
# @dataclass
# class PlanStep:
#     type: Literal["think","tool","code"]
#     text: str
#     tool_name: Optional[str] = None
#     args: Optional[Dict[str,Any]] = None
#     is_final: bool = False
@dataclass
class PlanStep:
    type: Literal["think","tool","code"]
    text: str
    tool_name: Optional[str] = None
    args: Optional[Dict[str,Any]] = None
    is_final: bool = False
    children: List["PlanStep"] = field(default_factory=list)



class BaseReasoner(ABC):
    def __init__(self, llm: GPTModel, max_depth: int = 5):
        self.llm = llm
        self.max_depth = max_depth
        self.current_plan = None
        self.plan_history = []

    @abstractmethod
    def generate(self, query: str, context: Dict[str, Any]) -> PlanStep:
        pass

    def validate_step(self, step: PlanStep) -> bool:
        """Production-grade validation for each reasoning step"""
        validation_prompt = f"""
        Validate this reasoning step: {step.thought}
        Context: {json.dumps(step.state)}
        Respond with VALID or INVALID followed by reason.
        """
        
        response = self.llm.generate(validation_prompt, max_length=100)
        if "INVALID" in response[:10]:
            logger.error(f"Invalid step detected: {response}")
            return False
        return True

class TreeOfThoughtReasoner(BaseReasoner):
    def __init__(self, *args, candidate_count: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate_count = candidate_count

    def generate(self, query: str, context: Dict[str, Any]) -> PlanStep:
        root = PlanStep(
            thought=f"Initial query: {query}",
            type=ThoughtType.OBSERVATION,
            state=context,
        )
        self.plan_history = []
        return self._expand_node(root, depth=0)

    def _expand_node(self, node: PlanStep, depth: int) -> PlanStep:
        if depth >= self.max_depth:
            return node

        # Generate candidate thoughts
        candidates = self._generate_candidates(node)
        validated = [c for c in candidates if self.validate_step(c)]
        
        # Select best candidate using LLM ranking
        best = self._select_best_candidate(validated)
        
        if best:
            best.children = []
            for candidate in validated:
                if candidate != best:
                    best.children.append(self._expand_node(candidate, depth+1))
            return best
        return node

    def _generate_candidates(self, node: PlanStep) -> List[PlanStep]:
        prompt = f"""
        Current state: {node.state}
        Previous thought: {node.thought}
        
        Generate {self.candidate_count} possible next thoughts. Format as JSON:
        {{
            "thoughts": [
                {{
                    "description": "thought description",
                    "type": "thought_type",
                    "action": {{"type": "action_type", "params": {{}}}}
                }}
            ]
        }}
        """
        
        response = self.llm.generate(prompt, temperature=0.7)
        try:
            data = json.loads(response)
            return [
                PlanStep(
                    thought=item["description"],
                    type=ThoughtType(item["type"]),
                    state=self._update_state(node.state, item["action"]),
                    parent=node
                ) for item in data["thoughts"]
            ]
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse candidates: {e}")
            return []

class ReActReasoner(BaseReasoner):
    def generate(self, query: str, context: Dict[str, Any]) -> PlanStep:
        current_step = PlanStep(
            thought=f"Initial query: {query}",
            type=ThoughtType.OBSERVATION,
            state=context
        )
        
        for _ in range(self.max_depth):
            # Generate thought
            thought_prompt = self._create_react_prompt(current_step)
            response = self.llm.generate(thought_prompt)
            
            # Parse action
            action = self._parse_action(response)
            if not action:
                break
                
            # Execute action
            result = self._execute_action(action)
            
            # Update state
            new_state = current_step.state.copy()
            new_state.update(result)
            
            # Create new step
            current_step = PlanStep(
                thought=response,
                type=ThoughtType.ACTION,
                state=new_state,
                parent=current_step
            )
        
        return current_step

class ReasoningAgent:
    def __init__(self, 
                 llm: GPTModel,
                 strategy: str = "tot",
                 code_validation: bool = True):
        self.llm = llm
        self.code_validation = code_validation
        
        self.reasoners = {
            "tot": TreeOfThoughtReasoner(llm=llm),
            "react": ReActReasoner(llm=llm)
        }
        
        if strategy not in self.reasoners:
            raise ValueError(f"Invalid strategy: {strategy}")
            
        self.reasoner = self.reasoners[strategy]
        self.active_plan = None

    def process_query(self, query: str, context: Dict = None) -> dict:
        context = context or {}
        plan = self.reasoner.generate(query, context)
        
        if self.code_validation:
            self._validate_code_actions(plan)
            
        self.active_plan = plan
        return self._format_output(plan)

    def _validate_code_actions(self, plan: PlanStep):
        current = plan
        while current:
            # if current.type == ThoughtType.ACTION:
            #     if self._is_code_action(current):
            #         if not self._safe_validate_code(current.state.get('code')):
            #             raise InvalidActionError("Code validation failed")
            if current.type == "code":
                code_to_run = current.text
                if not self._safe_validate_code(code_to_run):
                    raise InvalidActionError("Code validation failed")
            current = current.parent

    def _safe_validate_code(self, code: str) -> bool:
        """Production-grade code validation"""
        # Implement actual code validation/sandboxing here
        return True  # Placeholder

class InvalidActionError(Exception):
    pass