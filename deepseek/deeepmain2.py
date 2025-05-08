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

class InteractionLogger:
    """Simple JSONL logger for interactions to support continual learning."""
    def __init__(self, path: str = "interaction_logs.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]):
        record["timestamp"] = datetime.datetime.utcnow().isoformat()
        with open(self.path, 'a') as f:
            f.write(json.dumps(record) + "\n")

class InteractionDataset(Dataset):
    """Dataset wrapping logged interactions for fine-tuning."""
    def __init__(self, logs: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.samples = logs
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
        return {'input_ids': enc.input_ids.squeeze(0), 'attention_mask': enc.attention_mask.squeeze(0),
                'labels': tgt.input_ids.squeeze(0)}

class DeepSeekCore:
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize core components
        self.model_manager = ModelManager(self.config)
        self.task_router = TaskRouter(self.config)
        self.fusion_engine = HybridFusion(['moe', 'retnet', 'rwkv'], self.config)
        
        # Initialize autonomous systems
        self.observer = ApplicationObserver()
        self.executor = SafeExecutor()
        self.safety = ConstitutionalAI()
        self.approval = HumanApproval()
        self.self_model = SelfModel()
        self.logger = InteractionLogger(self.config.get('logging', {}).get('interaction_log', 'interaction_logs.jsonl'))
        
        # Set up initial model
        self.current_model = None
        self.current_model_type = None
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

            # 2. Task classification and model selection
            task_type = self.task_router.classify_task(text)
            model_type = self._map_task_to_model(task_type)

            # 3. Model management
            if model_type != self.current_model_type:
                self._switch_model(model_type)

            # 4. Reasoning and generation
            with torch.inference_mode():
                if self.config['system'].get('use_fusion', False):
                    outputs = self._fusion_generation(state)
                else:
                    outputs = self.current_model.generate(text)

            # 5. Safety validation
            if not self.safety.validate(outputs):
                raise SecurityError("Output failed safety check")

            # 6. Execution planning
            execution_plan = self._create_execution_plan(outputs)

            # 7. Human approval check
            if self._requires_approval(execution_plan):
                self._handle_approval(execution_plan)

            # 8. Execution
            result = self.executor.execute(execution_plan)

            # 9. Self-model update
            self.self_model.update(result)

            # 10. Log interaction for continual learning
            self._log_interaction(text, outputs, result)

            return {"result": result, "model_used": self.current_model_type, "confidence": outputs.get('confidence', 1.0)}

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return self._handle_failure(e)

    def _switch_model(self, new_model_type: str):
        logger.info(f"Switching model from {self.current_model_type} to {new_model_type}")
        self.model_manager.load_model(new_model_type)
        self.current_model = self.model_manager.get_model(new_model_type)
        old = self.current_model_type
        self.current_model_type = new_model_type
        if old:
            self.model_manager.unload_model(old)

    def _fusion_generation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        model_outputs = {}
        for m in ['moe', 'retnet', 'rwkv']:
            model = self.model_manager.load_model(m)
            with torch.no_grad():
                model_outputs[m] = model.generate(state['text'])
        return self.fusion_engine.fuse_outputs(model_outputs)

    def _map_task_to_model(self, task_type: Any) -> str:
        mapping = self.config.get('task_model_mapping', {})
        return mapping.get(str(task_type), self.config['system'].get('fallback_model', 'moe'))

    def _create_execution_plan(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"steps": outputs.get('steps', []), "confidence": outputs.get('confidence', 1.0), "model_type": self.current_model_type}

    def _requires_approval(self, plan: Dict[str, Any]) -> bool:
        return any(step.get('requires_approval', False) for step in plan.get('steps', []))

    def _handle_approval(self, plan: Dict[str, Any]):
        token = self.approval.request_approval(str(plan), "Autonomous operation requiring approval")
        if not self.approval.validate_token(token):
            raise SecurityError("Required approval not granted")

    def _handle_failure(self, error: Exception) -> Dict[str, Any]:
        return {"error": str(error), "recovery_attempted": True, "fallback_output": "I encountered an error. Please try again."}

    def _log_interaction(self, text: str, outputs: Dict[str, Any], result: Any):
        record = {"input": text, "model": self.current_model_type, "outputs": outputs, "result": result}
        self.logger.log(record)

    def retrain(self, epochs: int = 1, batch_size: int = 8, lr: float = 5e-5):
        """Offline re-training loop using logged interactions"""
        # 1. Load logs
        logs = []
        if not self.logger.path.exists():
            logger.warning("No interaction logs found for retraining.")
            return
        with open(self.logger.path) as f:
            for line in f:
                logs.append(json.loads(line))
        # 2. For each model, fine-tune separately
        for model_key in ['moe', 'retnet', 'rwkv']:
            model = self.model_manager.get_model(model_key)
            tokenizer = getattr(model, 'tokenizer', None)
            if tokenizer is None:
                logger.warning(f"Model {model_key} has no tokenizer; skipping fine-tune.")
                continue
            dataset = InteractionDataset([r for r in logs if r['model']==model_key], tokenizer)
            if len(dataset)==0:
                logger.info(f"No logs for model {model_key}; skipping.")
                continue
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            model.train()
            for epoch in range(epochs):
                for batch in loader:
                    optimizer.zero_grad()
                    outputs = model(batch['input_ids'].to(model.device), attention_mask=batch['attention_mask'].to(model.device))
                    logits = outputs.logits  # assume model returns logits
                    loss = loss_fn(logits.view(-1, logits.size(-1)), batch['labels'].to(model.device).view(-1))
                    loss.backward()
                    optimizer.step()
                logger.info(f"Epoch {epoch+1}/{epochs} done for {model_key}")
            # 3. Save updated checkpoint
            new_ckpt = Path(self.logger.path).parent / f"{model_key}_finetuned.pt"
            model.save(new_ckpt)
            self.model_manager.update_checkpoint(model_key, str(new_ckpt))
            logger.info(f"Saved finetuned checkpoint for {model_key} to {new_ckpt}")


def main():
    deepseek = DeepSeekCore()
    count = 0
    interval = deepseek.config.get('system', {}).get('retrain_interval', 50)
    while True:
        try:
            input_data = {"text": input("User: ")}
            response = deepseek.process_input(input_data)
            print(f"DeepSeek: {response['result']}")
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
