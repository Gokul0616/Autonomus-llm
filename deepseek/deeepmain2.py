import torch
import logging
import yaml
import json
import datetime
from pathlib import Path
from typing import Dict, Any, List
from torch.utils.data import Dataset, DataLoader

# Import core components
from deepseek.architectures import MoE, RetNet, RWKV
from deepseek.brain import TaskRouter, ModelManager, HybridFusion
from deepseek.autonomous.perception.app_observer import ApplicationObserver
from deepseek.autonomous.actions.executor import SafeExecutor
from deepseek.autonomous.safety.approval import HumanApproval, ConstitutionalAI, SecurityError
from deepseek.autonomous.self_model.self_model import SelfModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("deepseek.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EWC:
    """Elastic Weight Consolidation regularizer"""
    def __init__(self, model, dataloader, device, lambda_coef=0.4):
        self.model = model
        self.device = device
        self.lambda_coef = lambda_coef
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._estimate_fisher(dataloader)

    def _estimate_fisher(self, loader):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for batch in loader:
            self.model.zero_grad()
            inputs = batch['input_ids'].to(self.device)
            masks = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            outputs = self.model(inputs, attention_mask=masks)
            log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
            loss = torch.nn.functional.nll_loss(
                log_probs.view(-1, log_probs.size(-1)), labels.view(-1), ignore_index=self.model.tokenizer.pad_token_id
            )
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.clone().pow(2)
            break  # single batch for efficiency
        fisher = {n: f / len(loader) for n, f in fisher.items()}
        return fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return self.lambda_coef * loss

class InteractionLogger:
    """Simple JSONL logger for interactions to support continual learning with feedback and rewards."""
    def __init__(self, path: str = "interaction_logs.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer: List[Dict[str, Any]] = []

    def log(self, record: Dict[str, Any]):
        record["timestamp"] = datetime.datetime.utcnow().isoformat()
        record.setdefault("confidence", None)
        record.setdefault("feedback", None)
        record.setdefault("reward", None)
        self.buffer.append(record)
        # write streaming log
        with open(self.path, 'a') as f:
            f.write(json.dumps(record) + "\n")

    def update_last(self, feedback: bool, reward: float = 0.0):
        if not self.buffer:
            return
        last = self.buffer[-1]
        last["feedback"] = feedback
        last["reward"] = reward
        # rewrite file
        lines = []
        with open(self.path) as f:
            lines = f.readlines()
        lines[-1] = json.dumps(last) + "\n"
        with open(self.path, 'w') as f:
            f.writelines(lines)

class InteractionDataset(Dataset):
    """Dataset wrapping logged interactions for fine-tuning."""
    def __init__(self, logs: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.samples = [r for r in logs if r.get('feedback') is not False]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        text = rec['input']
        target = rec['result']
        enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        tgt = self.tokenizer(target, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {'input_ids': enc.input_ids.squeeze(0),
                'attention_mask': enc.attention_mask.squeeze(0),
                'labels': tgt.input_ids.squeeze(0)}

class DeepSeekCore:
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)
        # Core components
        self.model_manager = ModelManager(self.config)
        self.task_router = TaskRouter(self.config)
        self.fusion_engine = HybridFusion(['moe', 'retnet', 'rwkv'], self.config)
        # Autonomous systems
        self.observer = ApplicationObserver()
        self.executor = SafeExecutor()
        self.safety = ConstitutionalAI()
        self.approval = HumanApproval()
        self.self_model = SelfModel()
        self.logger = InteractionLogger(self.config.get('logging', {}).get('interaction_log', 'interaction_logs.jsonl'))
        # Setup
        self.current_model = None
        self.current_model_type = None
        self.ewc_modules: Dict[str, EWC] = {}
        self._load_initial_model()

    def _load_config(self, path: str) -> Dict[str, Any]:
        config_file = Path(__file__).parent / path
        with open(config_file) as f:
            return yaml.safe_load(f)

    def _load_initial_model(self):
        default = self.config.get('system', {}).get('default_model', 'moe')
        self.current_model = self.model_manager.load_model(default)
        self.current_model_type = default
        logger.info(f"Initialized with {default.upper()} model")

    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 1. Perception
            state = self.observer.capture_state(input_data)
            text = state['text']
            # 2. Routing
            task_type = self.task_router.classify_task(text)
            model_type = self._map_task_to_model(task_type)
            # 3. Switch if needed
            if model_type != self.current_model_type:
                self._switch_model(model_type)
            # 4. Inference
            with torch.inference_mode():
                if self.config['system'].get('use_fusion', False):
                    outputs = self._fusion_generation(state)
                else:
                    outputs = self.current_model.generate(text)
            # record confidence
            outputs['confidence'] = outputs.get('confidence', None)
            # 5. Safety
            if not self.safety.validate(outputs):
                raise SecurityError("Output failed safety check")
            # 6. Plan
            execution_plan = self._create_execution_plan(outputs)
            # 7. Approval
            if self._requires_approval(execution_plan):
                self._handle_approval(execution_plan)
            # 8. Execute
            result = self.executor.execute(execution_plan)
            # 9. Self-model
            self.self_model.update(result)
            # 10. Log
            self._log_interaction(text, outputs, result)
            # 11. Streaming update (mini-batch)
            self._online_update()
            return {"result": result, "model_used": self.current_model_type, "confidence": outputs['confidence']}
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return self._handle_failure(e)

    def _switch_model(self, new_model_type: str):
        logger.info(f"Switching model from {self.current_model_type} to {new_model_type}")
        self.model_manager.load_model(new_model_type)
        self.current_model = self.model_manager.get_model(new_model_type)
        self.current_model_type = new_model_type
        # setup EWC for new model if not exists
        self.ewc_modules[new_model_type] = None

    def _fusion_generation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        outs = {}
        for m in ['moe', 'retnet', 'rwkv']:
            model = self.model_manager.load_model(m)
            with torch.no_grad(): outs[m] = model.generate(state['text'])
        return self.fusion_engine.fuse_outputs(outs)

    def _map_task_to_model(self, task_type: Any) -> str:
        return self.config.get('task_model_mapping', {}).get(str(task_type),
               self.config['system'].get('fallback_model', 'moe'))

    def _create_execution_plan(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"steps": outputs.get('steps', []), "confidence": outputs['confidence'], "model_type": self.current_model_type}

    def _requires_approval(self, plan: Dict[str, Any]) -> bool:
        return any(step.get('requires_approval', False) for step in plan.get('steps', []))

    def _handle_approval(self, plan: Dict[str, Any]):
        token = self.approval.request_approval(str(plan), "Autonomous operation requiring approval")
        if not self.approval.validate_token(token): raise SecurityError("Required approval not granted")

    def _handle_failure(self, error: Exception) -> Dict[str, Any]:
        return {"error": str(error), "recovery_attempted": True, "fallback_output": "I encountered an error."}

    def _log_interaction(self, text: str, outputs: Dict[str, Any], result: Any):
        rec = {"input": text, "model": self.current_model_type, "outputs": outputs, "result": result}
        self.logger.log(rec)

    def _online_update(self):
        """Perform a quick mini-batch update on the most recent interaction."""
        # load last record
        logs = []
        with open(self.logger.path) as f:
            for line in f: logs.append(json.loads(line))
        last = logs[-1]
        tokenizer = getattr(self.current_model, 'tokenizer', None)
        if tokenizer is None or last.get('feedback') is False:
            return
        ds = InteractionDataset([last], tokenizer)
        dl = DataLoader(ds, batch_size=1)
        self.current_model.train()
        optimizer = torch.optim.Adam(self.current_model.parameters(), lr=self.config['system'].get('online_lr', 1e-5))
        for batch in dl:
            optimizer.zero_grad()
            out = self.current_model(batch['input_ids'].to(self.current_model.device),
                                     attention_mask=batch['attention_mask'].to(self.current_model.device))
            logits = out.logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch['labels'].view(-1).to(self.current_model.device),
                ignore_index=tokenizer.pad_token_id
            )
            # add EWC penalty
            if self.ewc_modules.get(self.current_model_type) is None:
                # initialize EWC if missing
                self.ewc_modules[self.current_model_type] = EWC(
                    self.current_model,
                    dl,
                    device=self.current_model.device,
                    lambda_coef=self.config['system'].get('ewc_lambda', 0.4)
                )
            ewc = self.ewc_modules[self.current_model_type]
            loss += ewc.penalty(self.current_model)
            loss.backward()
            optimizer.step()
        self.current_model.eval()

    def retrain(self, epochs: int = 1, batch_size: int = 8, lr: float = 5e-5):
        """Offline batch re-training using interaction logs and EWC."""
        # load logs
        logs = []
        if not self.logger.path.exists():
            logger.warning("No logs for retraining.")
            return
        with open(self.logger.path) as f:
            logs = [json.loads(line) for line in f]
        for model_key in ['moe', 'retnet', 'rwkv']:
            # filter only positive-feedback
            filtered = [r for r in logs if r['model']==model_key and r.get('feedback')]
            if not filtered:
                logger.info(f"No positive logs for {model_key}, skip.")
                continue
            model = self.model_manager.get_model(model_key)
            tokenizer = getattr(model, 'tokenizer', None)
            ds = InteractionDataset(filtered, tokenizer)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            # initialize EWC
            ewc = EWC(model, dl, device=model.device, lambda_coef=self.config['system'].get('ewc_lambda', 0.4))
            model.train()
            for epoch in range(epochs):
                for batch in dl:
                    optimizer.zero_grad()
                    out = model(batch['input_ids'].to(model.device), attention_mask=batch['attention_mask'].to(model.device))
                    logits = out.logits
                    loss = loss_fn(logits.view(-1, logits.size(-1)), batch['labels'].view(-1).to(model.device))
                    loss += ewc.penalty(model)
                    loss.backward()
                    optimizer.step()
                logger.info(f"Batch retrain epoch {epoch+1}/{epochs} for {model_key}")
            # save and update
            new_ckpt = Path(self.logger.path).parent / f"{model_key}_finetuned.pt"
            model.save(new_ckpt)
            self.model_manager.update_checkpoint(model_key, str(new_ckpt))
            logger.info(f"Saved retrained {model_key} to {new_ckpt}")


def main():
    deepseek = DeepSeekCore()
    count = 0
    interval = deepseek.config['system'].get('retrain_interval', 50)
    while True:
        try:
            txt = input("User: ")
            # after response, ask feedback
            response = deepseek.process_input({"text": txt})
            print(f"DeepSeek: {response['result']}")
            # collect feedback
            fb = input("Feedback (y/n)? ").strip().lower() == 'y'
            reward = float(input("Reward (0.0-1.0)? ").strip() or 0.0)
            deepseek.logger.update_last(fb, reward)
            count += 1
            if count >= interval:
                deepseek.retrain(
                    epochs=deepseek.config['system'].get('retrain_epochs', 1),
                    batch_size=deepseek.config['system'].get('retrain_batch_size', 8),
                    lr=deepseek.config['system'].get('retrain_lr', 5e-5)
                )
                count = 0
        except KeyboardInterrupt:
            logger.info("Shutting down safely...")
            deepseek.model_manager.cleanup()
            torch.cuda.empty_cache()
            break

if __name__ == "__main__":
    main()
