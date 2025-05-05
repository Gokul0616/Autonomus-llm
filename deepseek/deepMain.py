
# deepMain.py
import os
import json
import torch
from datetime import datetime

# --- Transformer, Tokenizer & Training ---
from deepseek.deepmodel import GPTModel, train_model, evaluate_model
from BPETokenizer import CustomBPETokenizer

# --- Monitoring & Versioning ---
from model_versioning import ModelVersioner
from monitoring import ModelMonitor

# --- Memory & Self-Model ---
from autonomous.Memory.memory_store import GPTMemoryStore
from autonomous.self_model.self_model import SelfModel
from autonomous.self_model.narrative import NarrativeGenerator

# --- Reasoning & Self-Improvement ---
from deepseek.reasoning import ReasoningAgent, InvalidActionError, TreeOfThoughtReasoner
from self_improvement import CodeValidator, SelfImprovementEngine
from reward_model import RewardModel

# --- Autonomous Agent & Tools ---
from autonomous.agent.agent import AutonomousAgent
from autonomous.tools.software import ShellTool, DesktopAutomationTool, RestAPITool
from autonomous.tools.hardware import LEDTool, ServoTool, SensorTool
from autonomous.perception.app_observer import ApplicationObserver
from autonomous.actions.executor import ActionExecutor, SafeExecutor
from autonomous.safety.approval import HumanApproval, ConstitutionalAI

# --- Helper: Ensure training and tokenizer ---
def ensure_training_and_tokenizer(model_path="transformer_model.pth", tok_path="tokenizer.json"):
    # Train model if checkpoint missing
    if not os.path.exists(model_path):
        print(f"[Setup] Training Base Model → {model_path}")
        # Example: load data and call train_model
        # dataset, tokenizer = prepare_dataset_and_tokenizer()
        tokenizer = CustomBPETokenizer(vocab_size=10000)
        # tokenizer.build_vocab(texts)
        model = GPTModel(tokenizer=tokenizer)
        train_model(model, dataset=None, tokenizer=tokenizer)
        torch.save(model.state_dict(), model_path)
    else:
        print(f"[Setup] Found model checkpoint: {model_path}")
    # Ensure tokenizer exists
    if not os.path.exists(tok_path):
        print(f"[Setup] Saving tokenizer → {tok_path}")
        # tokenizer.save(tok_path)  # implement save in BPETokenizer
        with open(tok_path, 'w') as f:
            json.dump(CustomBPETokenizer().token_to_id, f)

# --- Initialize components once ---
def initialize_components(use_improved=False):
    # Load tokenizer
    tokenizer = CustomBPETokenizer()
    tokenizer.load("tokenizer.json")

    # Load model checkpoint
    path = "improved_model.pth" if use_improved else "transformer_model.pth"
    model = GPTModel(tokenizer=tokenizer)
    model.load_state_dict(torch.load(path))

    # Monitoring and rate limiting
    monitor = ModelMonitor()
    model.generate = monitor.track(model.generate)
    model.monitor = monitor

    # Versioning
    versioner = ModelVersioner()

    # Memory & Self-Model
    memory = GPTMemoryStore(model=model, tokenizer=tokenizer)
    self_model = SelfModel()
    narrator = NarrativeGenerator(model)

    # Reward model
    reward_model = RewardModel(model)

    # Reasoning & planning
    reasoner = TreeOfThoughtReasoner(llm=model)
    reason_agent = ReasoningAgent(llm=model, strategy="tot", code_validation=True)

    # Tools
    tools = {
        "shell": ShellTool(),
        "desktop": DesktopAutomationTool(),
        "http": RestAPITool(),
        "code_exec": CodeValidator(),
        "led": LEDTool(pin=17),
        "servo": ServoTool(pin=18),
        "sensor": SensorTool(pin=27),
    }
    auto_agent = AutonomousAgent(llm=model, memory=memory, tools=tools, reasoner=reasoner)

    # Perception, Execution & Safety
    observer = ApplicationObserver()
    executor = SafeExecutor()
    approval = HumanApproval()
    constitutional = ConstitutionalAI()

    # Self-improvement
    improver = SelfImprovementEngine(model, CodeValidator())

    return {
        "model": model,
        "tokenizer": tokenizer,
        "monitor": monitor,
        "versioner": versioner,
        "memory": memory,
        "self_model": self_model,
        "narrator": narrator,
        "reason_agent": reason_agent,
        "auto_agent": auto_agent,
        "observer": observer,
        "executor": executor,
        "approval": approval,
        "constitutional": constitutional,
        "improver": improver,
        "reward_model": reward_model
    }

# --- Unified Autonomous Loop ---
def unified_loop(comps):
    c = comps
    while True:
        # Observe
        state = c["observer"].capture_state()

        # Memory recall
        context_mem = c["memory"].query(json.dumps(state))

        # Planning with reasoning agent
        plan = c["reason_agent"].process_query(
            "Suggest improvements based on state", 
            context={"state": state, "memory": context_mem, "self": c["self_model"].data}
        )

        # Pre-execution introspection
        introspect = c["model"].generate(
            f"Given self-model {c['self_model'].data} and plan {plan}, any issues?"
        )
        plan = c["reason_agent"].process_query(
            f"Revise plan: {introspect}",
            context={"state": state, "self": c["self_model"].data}
        )

        # Safety & approval
        if not c["constitutional"].validate(plan):
            continue
        if not c["approval"].check(plan):
            continue

        # Execution
        outcome = c["executor"].execute(plan)

        # Post-execution reflection
        reflection = c["model"].generate(f"Reflect on outcome: {outcome}")
        c["self_model"].update(reflection)

        # Store memory
        c["memory"].write(state, plan, outcome, reflection)

        # Narrative generation
        if len(c["self_model"].data.get("reflections", [])) % 10 == 0:
            narrative = c["narrator"].generate(
                c["self_model"].data["reflections"][-10:]
            )
            c["self_model"].data.setdefault("narrative", []).append(narrative)
            c["self_model"].save()

        # Self-improvement
        improvement = c["improver"].generate_improvement("Improve next action loop")
        c["improver"].learn_from_experience(improvement)

        # Versioning
        c["versioner"].save(c["model"])

        # Online continuous training
        c["model"].continuous_train([{
            "state": state,
            "action": plan,
            "reward": c["reward_model"].compute_reward(improvement.get("validation", {}), improvement.get("execution_time", 0))
        }])

if __name__ == "__main__":
    # Setup training & tokenizer
    ensure_training_and_tokenizer()

    # Initialize all components
    comps = initialize_components(use_improved=False)

    # Start monitoring server
    comps["monitor"].start_server(port=9090)

    # Run unified loop
    unified_loop(comps)
