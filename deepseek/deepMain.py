# main.py
from reasoning import ReasoningAgent, InvalidActionError
from self_improvement import CodeValidator, SelfImprovementEngine
import torch
import json
from deepseek.utils.tokenizerUtils import load_tokenizer
from deepmodel import GPTModel

def initialize_agent(improved_model_path=None):
    tokenizer = load_tokenizer()
    model = GPTModel(tokenizer=tokenizer)
    
    # Load either improved or base model
    model_path = improved_model_path if improved_model_path else "transformer_model.pth"
    model.load_state_dict(torch.load(model_path))
    
    return ReasoningAgent(
        llm=model,
        strategy="tot",
        code_validation=True
    )

def main_loop(use_improved=False):
    try:
        # Initialize components
        improved_path = "improved_model.pth" if use_improved else None
        agent = initialize_agent(improved_path)
        validator = CodeValidator(use_docker=True)
        improvement_engine = SelfImprovementEngine(agent.llm, validator)

        # Self-improvement phase
        improvement_task = (
            "Optimize the transformer architecture using pytorch's flash attention "
            "and implement beam search decoding"
        )
        improvement_result = improvement_engine.generate_improvement(improvement_task)
        
        if improvement_result["validation"]["test_passed"]:
            print("Model successfully self-improved! Applying changes...")
            torch.save(agent.llm.state_dict(), "improved_model.pth")
        else:
            print(f"Self-improvement failed: {improvement_result['validation']['error']}")

        # Process user query with current model
        query_result = agent.process_query(
            "Implement a secure login system in Python with SQL injection protection",
            context={"requirements": ["password hashing", "sql injection"]}
        )
        
        print("\nFinal Implementation Plan:")
        print(json.dumps(query_result, indent=2))

    except InvalidActionError as e:
        print(f"Security block: {e}")
    except Exception as e:
        print(f"Critical error: {str(e)}")

if __name__ == "__main__":
    # First run with base model
    print("Monitoring Start in port 9090")
    GPTModel.monitor.start_server(port=9090)
    print("=== Initial Run with Base Model ===")
    main_loop(use_improved=False)
    
    # Subsequent runs can use improved model
    print("\n=== Follow-up Run with Improved Model ===")
    main_loop(use_improved=True)