import time
import json
from agent import ToolUseAgent

def run_evaluation():
    print("🚀 Starting Agent Evaluation Framework...\n")
    
    with open("eval_dataset.json", "r") as f:
        test_cases = json.load(f)
        
    agent = ToolUseAgent()
    results = []

    for test in test_cases:
        test_id = test["id"]
        prompt = test["prompt"]
        expected_tools = set(test["expected_tools"])
        
        print(f"Running [{test_id}]: {test['description']}...")
        
        start_time = time.time()
        try:
            output = agent.run(prompt, max_steps=5)
            latency = round(time.time() - start_time, 2)
            
            # Extract tools actually called during execution
            called_tools = set([call["tool"] for call in output.get("tool_calls", [])])
            
            # Metric Calculations: Precision & Recall for Tool Selection
            true_positives = len(expected_tools.intersection(called_tools))
            precision = true_positives / len(called_tools) if called_tools else (1.0 if not expected_tools else 0.0)
            recall = true_positives / len(expected_tools) if expected_tools else 1.0
            
            passed_tools = (expected_tools == called_tools)
            
            results.append({
                "id": test_id,
                "passed_tool_selection": passed_tools,
                "expected_tools": list(expected_tools),
                "called_tools": list(called_tools),
                "precision": round(precision, 2),
                "recall": round(recall, 2),
                "latency_seconds": latency,
                "provider_used": output.get("provider", "Unknown"),
                "turns_taken": output.get("steps", 1)
            })
            
            status = "✅ PASS" if passed_tools else "❌ FAIL"
            print(f"   {status} | Latency: {latency}s | Provider: {output.get('provider')} | Called: {list(called_tools)}")
            
        except Exception as e:
            print(f"   ❌ ERROR executing test case: {e}")
            results.append({
                "id": test_id,
                "error": str(e),
                "passed_tool_selection": False
            })

    print("\n" + "="*50)
    print("📊 EVALUATION SUMMARY REPORT")
    print("="*50)
    
    total = len(results)
    passed = sum(1 for r in results if r.get("passed_tool_selection"))
    avg_latency = round(sum(r.get("latency_seconds", 0) for r in results) / total, 2)
    
    print(f"Total Test Cases : {total}")
    print(f"Passed          : {passed} / {total} ({(passed/total)*100:.1f}%)")
    print(f"Avg Latency     : {avg_latency} seconds")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()