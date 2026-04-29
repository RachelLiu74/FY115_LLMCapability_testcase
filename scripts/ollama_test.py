import ollama
import json
import time
import platform
from datetime import datetime

# CONFIGURATION: Define your hardware and test cases
PLATFORM_NAME = "A6000-Ada" #"GTX_1080Ti"  # Change to "GB200" or "MacBook_M2" depending on machine
MODELS_TO_TEST =  ["gpt-oss:20b"]#[gemma3:4b, "llama3:8b", "mistral:7b"]
TEST_CASES = [
    {"name": "Short QA", "prompt": "Explain what a neural network is in one sentence."},
    {"name": "Creative Writing", "prompt": "Write a 200-word story about a robot learning to cook."},
    {"name": "Code Gen", "prompt": "Write a Python script to scrape a website using BeautifulSoup."}
]

def run_benchmark():
    results = []
    
    for model in MODELS_TO_TEST:
        print(f"\n🚀 Benchmarking Model: {model}")
        
        for case in TEST_CASES:
            print(f"  - Running case: {case['name']}...", end="", flush=True)
            
            # Record start for manual wall-clock time if needed
            start_time = time.time()
            
            # API Call with performance metrics enabled (stream=False)
            response = ollama.generate(
                model=model,
                prompt=case['prompt'],
                stream=False
            )
            
            # Calculate Tokens Per Second (TPS)
            # Ollama returns durations in nanoseconds (10^9)
            eval_count = response.get('eval_count', 0)
            eval_duration_ns = response.get('eval_duration', 1)
            tps = (eval_count / eval_duration_ns) * 1e9
            
            benchmark_entry = {
                "timestamp": datetime.now().isoformat(),
                "platform": PLATFORM_NAME,
                "model": model,
                "test_case": case['name'],
                "metrics": {
                    "total_duration_sec": response.get('total_duration', 0) / 1e9,
                    "load_duration_sec": response.get('load_duration', 0) / 1e9,
                    "tokens_per_second": round(tps, 2),
                    "eval_count": eval_count,
                    "prompt_eval_count": response.get('prompt_eval_count', 0)
                }
            }
            results.append(benchmark_entry)
            print(f" Done! ({round(tps, 2)} tok/s)")

    save_results(results)

def save_results(new_data):
    filename = "/media/r300/1T/A30335/agentMCP/tester202601/gpt_oss20b_json/A6000-Ada-gptoss20b-ollama_performance_memory.json"
    try:
        with open(filename, "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []
    
    history.extend(new_data)
    
    with open(filename, "w") as f:
        json.dump(history, f, indent=4)
    print(f"\n✅ Results memorized to {filename}")

if __name__ == "__main__":
    run_benchmark()