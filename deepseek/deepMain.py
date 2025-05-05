# deepMain.py
import os
import json
import torch
from datetime import datetime

# --- Core Model & Tokenizer ---
from deepseek.deepmodel import GPTModel, train_model, evaluate_model
from BPETokenizer import CustomBPETokenizer

# --- Memory & Self-Model ---
from autonomous.Memory.memory_store import GPTMemoryStore
from autonomous.self_model.self_model import SelfModel
from autonomous.self_model.narrative import NarrativeGenerator

# --- Reasoning & Self-Improvement ---
from deepseek.reasoning import ReasoningAgent, InvalidActionError, TreeOfThoughtReasoner
from self_improvement import CodeValidator, SelfImprovementEngine
from reward_model import RewardModel

# --- Tools & Execution ---
from autonomous.agent.agent import AutonomousAgent
from autonomous.tools.software import ShellTool, DesktopAutomationTool, RestAPITool
from autonomous.tools.hardware import LEDTool, ServoTool, SensorTool
from autonomous.perception.app_observer import ApplicationObserver
from autonomous.actions.executor import ActionExecutor, SafeExecutor
from autonomous.safety.approval import HumanApproval, ConstitutionalAI

# --- Monitoring & Versioning ---
from model_versioning import ModelVersioner
from monitoring import ModelMonitor

# Startup: ensure model exists (Training & Eval)
def ensure_model(path="transformer_model.pth"):
    if not os.path.exists(path):
        print("[Training] No checkpoint found, training model...")
        # Load data, train, save
        train_model_checkpoint(path)
    else:
        print(f"[Startup] Found checkpoint: {path}")

# Example wrapper to train and save
def train_model_checkpoint(path):
    # Placeholder: implement your data loading here
    # texts = ...
    tokenizer = CustomBPETokenizer(vocab_size=10000)
    # tokenizer.build_vocab(texts)
    model = GPTModel(tokenizer=tokenizer)
    # train_model(model, dataset, tokenizer)
    torch.save(model.state_dict(), path)
    with open("tokenizer.json", "w") as f:
        json.dump(tokenizer.token_to_id, f)
    print(f"[Training] Saved checkpoint to {path}")

# Initialize agent and components
def initialize_components(use_improved=False):
    # Load tokenizer + model
    tokenizer = CustomBPETokenizer()
    tokenizer.load("tokenizer.json")
    model_path = "improved_model.pth" if use_improved else "transformer_model.pth"
    model = GPTModel(tokenizer=tokenizer)
    model.load_state_dict(torch.load(model_path))

    # Apply monitoring
    monitor = ModelMonitor()
    model.generate = monitor.track(model.generate)
    model.monitor = monitor

    # Versioning
    versioner = ModelVersioner()

    # Memory & Self
    memory_store = GPTMemoryStore(model=model, tokenizer=tokenizer)
    self_model = SelfModel()
    narrative_gen = NarrativeGenerator(model)

    # Reasoning agent
    reasoner = TreeOfThoughtReasoner(llm=model)
    agent = ReasoningAgent(llm=model, strategy="tot", code_validation=True)
    reward_model = RewardModel(model)

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
    auto_agent = AutonomousAgent(
        llm=model,
        memory=memory_store,
        tools=tools,
        reasoner=reasoner
    )

    # Observer & Executors & Safety
    observer = ApplicationObserver()
    safe_exec = SafeExecutor()
    human_approval = HumanApproval()
    constitutional = ConstitutionalAI()
    self_improve = SelfImprovementEngine(model, CodeValidator())

    return {
        "model": model,
        "tokenizer": tokenizer,
        "monitor": monitor,
        "versioner": versioner,
        "memory": memory_store,
        "self_model": self_model,
        "narrator": narrative_gen,
        "agent": agent,
        "auto_agent": auto_agent,
        "observer": observer,
        "executor": safe_exec,
        "approval": human_approval,
        "constitutional": constitutional,
        "improve": self_improve,
        "reward_model": reward_model
    }

# Unified Autonomous Loop (observe → plan → introspect → act → reflect → learn)
def autonomous_loop(components):
    c = components
    while True:
        # 1. Observe
        state = c["observer"].capture_state()
        # 2. Memory recall
        past = c["memory"].query(json.dumps(state))
        # 3. Plan
        plan = c["agent"].process_query(
            "Analyze state and suggest improvements",
            context={"state": state, "memory": past, "self": c["self_model"].data}
        )
        # 4. Pre-exec introspection
        introspect = c["model"].generate(
            f"Given self {c['self_model'].data} and plan {plan}, any issues?"
        )
        plan = c["agent"].process_query(
            f"Revise plan based on introspection: {introspect}",
            context={"state": state, "self": c["self_model"].data}
        )
        # 5. Safety & Approval
        if not c["constitutional"].validate(plan):
            continue
        if not c["approval"].check(plan):
            continue
        # 6. Execute
        outcome = c["executor"].execute(plan)
        # 7. Post-exec reflection
        reflection = c["model"].generate(f"Reflect on the outcome: {outcome}")
        c["self_model"].update(reflection)
        # 8. Memory store
        c["memory"].write(state, plan, outcome, reflection)
        # 9. Narrative
        if len(c["self_model"].data.get("reflections", [])) % 10 == 0:
            narr = c["narrator"].generate(c["self_model"].data["reflections"][-10:])
            c["self_model"].data.setdefault("narrative", []).append(narr)
            c["self_model"].save()
        # 10. Self-Improvement & Online Training
        val = c["improve"].generate_improvement("Optimize next steps")
        c["improve"].learn_from_experience(val)
        c["versioner"].save(c["model"])
        c["model"].continuous_train([{
            "state": state, 
            "action": plan,
            "reward": c["reward_model"].compute_reward(val.get("validation", {}), val.get("execution_time", 0))
        }])

if __name__ == "__main__":
    # 0. Ensure model trained
    ensure_model()
    # 1. Initialize all components
    comps = initialize_components()
    # 2. Start monitoring server
    comps["monitor"].start_server(port=9090)
    # 3. Launch unified autonomous loop
    autonomous_loop(comps)

