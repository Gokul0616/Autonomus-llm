# deepMain.py
import itertools
import os
import json
import torch
from datetime import datetime

# --- Transformer, Tokenizer & Training ---
from deepmodel import GPTModel, TextDataset, train_model, evaluate_model
from tokenizers.BPETokenizer import CustomBPETokenizer

# --- Monitoring & Versioning ---
from model_versioning import ModelVersioner
from monitoring import ModelMonitor

# --- Memory & Self-Model ---
from autonomous.Memory.memory_store import GPTMemoryStore
from autonomous.self_model.self_model import SelfModel
from autonomous.self_model.narrative import NarrativeGenerator

# --- Reasoning & Self-Improvement ---
from reasoning import ReasoningAgent, InvalidActionError, TreeOfThoughtReasoner
from self_improvement import CodeValidator, SelfImprovementEngine
from reward_model import RewardModel

# --- Autonomous Agent & Tools ---
from autonomous.agent.agent import AutonomousAgent
from autonomous.tools.software import ShellTool, DesktopAutomationTool, RestAPITool
from autonomous.tools.hardware import LEDTool, ServoTool, SensorTool
from autonomous.perception.app_observer import ApplicationObserver
from autonomous.actions.executor import ActionExecutor, SafeExecutor
from autonomous.safety.approval import HumanApproval, ConstitutionalAI
from datasets import  load_dataset
# --- Helper: Ensure training and tokenizer ---
from utils.tokenizerUtils import load_tokenizer as load_tokenizer_util

def ensure_training_and_tokenizer(
    model_path="transformer_model.pth",
    tok_path="tokenizer.json",
    vocab_size=10000,
    num_stream=200_000,
    block_size=128
):
    # 1) Prepare data streams
    print("[Setup] Streaming training data…")
    wiki_stream = load_dataset("wikipedia", "20220301.en", split="train", streaming=True, trust_remote_code=True)
    code_stream = load_dataset("code_search_net", "python", split="train", streaming=True, trust_remote_code=True)

    wiki_texts = list(itertools.islice((item["text"] for item in wiki_stream), num_stream))
    code_texts = list(itertools.islice((item["whole_func_string"] for item in code_stream), num_stream))
    texts = wiki_texts + code_texts

    # 2) Build or load tokenizer
    tokenizer = CustomBPETokenizer(vocab_size=vocab_size)
    if not os.path.exists(tok_path):
        print(f"[Setup] Building vocab ({vocab_size} tokens)…")
        tokenizer.build_vocab(texts)
        tokenizer.save(tok_path)
    else:
        print(f"[Setup] Loading existing tokenizer from {tok_path}…")
        tokenizer.load(tok_path)

    # 3) Instantiate dataset
    dataset = TextDataset(texts, tokenizer, block_size=block_size)

    # 4) Train model if needed
    if not os.path.exists(model_path):
        print(f"[Setup] Training model (this may take a while)…")
        model = GPTModel(tokenizer=tokenizer)
        train_model(model, dataset, tokenizer)
        torch.save(model.state_dict(), model_path)
        print(f"[Setup] Saved trained model to {model_path}")
    else:
        print(f"[Setup] Found existing model checkpoint at {model_path}")

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
    # ─── Step 0: Ensure we have a trained model + tokenizer ───
    ensure_training_and_tokenizer(
        model_path="transformer_model.pth",
        tok_path="tokenizer.json",
        vocab_size=10000,
        num_stream=200_000,
        block_size=128
    )

    # ─── Step 1: Initialize all the agent components ───
    comps = initialize_components(use_improved=False)

    # ─── Step 2: Start monitoring ───
    comps["monitor"].start_server(port=9090)

    # ─── Step 3: Enter your unified autonomous loop ───
    unified_loop(comps)