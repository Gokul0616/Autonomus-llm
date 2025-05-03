from typing import List, Any, Dict
from deepmodel import GPTModel
from reasoning import PlanStep, BaseReasoner
from autonomous.Memory.memory_store import GPTMemoryStore
from tools.software import ShellTool, DesktopAutomationTool, RestAPITool
from tools.hardware import LEDTool, ServoTool, SensorTool
import BPETokenizer

class AutonomousAgent:
    def __init__(
        self,
        llm: GPTModel,
        memory: GPTMemoryStore,
        tools: Dict[str, Any],
        reasoner: BaseReasoner
    ):
        """
        llm       : Your language model instance
        memory    : A GPTMemoryStore instance for persistent memory
        tools     : Dictionary of tool_name -> tool instance
        reasoner  : An instance of a BaseReasoner (e.g., TreeOfThoughtReasoner)
        """
        self.llm       = llm
        self.memory    = memory
        self.tools     = tools
        self.reasoner  = reasoner

    def run(self, goal: str, max_iters: int = 5):
        # 1. seed context from memory
        context = self.memory.query(goal)

        for iteration in range(max_iters):
            # 2. PLAN
            plan: PlanStep = self.reasoner.generate(goal, context)

            # 3. EXECUTE each step in a flattened plan
            for step in self._flatten(plan):
                if step.type == "code":
                    result = self.tools["code_exec"].run(step.text)
                elif step.type == "tool" and step.tool_name:
                    result = self.tools[step.tool_name].run(**(step.args or {}))
                else:
                    result = {"info": step.text}

                # 4. LEARN: write every result back into memory
                record = f"Step: {step.type} → {step.text} | Result: {result}"
                self.memory.write(record)
                context.append(record)

                if step.is_final:
                    print("✅ Goal reached.")
                    return

        print("⚠️ Max iterations reached.")
    
    def _flatten(self, root: PlanStep) -> List[PlanStep]:
        """Preorder traversal to turn a tree of PlanSteps into a list."""
        out = [root]
        for child in getattr(root, "children", []) or []:
            out += self._flatten(child)
        return out
