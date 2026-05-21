#!/usr/bin/env python3
"""
MI300X Performance Benchmark Script
Tests all available Ollama models on AMD Instinct MI300X (192GB VRAM, ROCm 7.1.1)
Measures: CONTEXT WINDOW, PREFILL, DECODE, TOKEN/S, WATT/TOKEN, INPUT/OUTPUT, Q&A similarity
"""

import ollama
import json
import time
import os
from datetime import datetime
from difflib import SequenceMatcher

# ===== CONFIGURATION =====
PLATFORM_NAME = "MI300X"
PLATFORM_TDP_W = 750  # AMD Instinct MI300X TDP ~750W

MODELS_TO_TEST = [
    "medgemma:latest",   # 4.3B - smallest
    "qwen3:4b",          # 4B
    "gpt-oss:20b",       # 20B
    "gemma4:26b",        # 26B
    "gemma4:31b",        # 31B (quantized from larger)
    "qwen3:32b",         # 32B
    "llama3.1:70b",      # 70B - largest
]

# ICOPE-related Q&A test cases with reference answers for similarity scoring
TEST_CASES = [
    {
        "name": "Short QA",
        "prompt": "Explain what a neural network is in one sentence.",
        "reference": "A neural network is a computational model inspired by the human brain that consists of interconnected nodes organized in layers to process and learn patterns from data."
    },
    {
        "name": "Creative Writing",
        "prompt": "Write a 200-word story about a robot learning to cook.",
        "reference": None  # No reference for creative tasks
    },
    {
        "name": "Code Gen",
        "prompt": "Write a Python script to scrape a website using BeautifulSoup.",
        "reference": None
    },
    {
        "name": "ICOPE Nursing QA",
        "prompt": "請用繁體中文說明ICOPE評估中，認知功能篩檢的標準流程與評分方式。",
        "reference": "ICOPE認知功能篩檢使用簡短認知評估工具，包含時間定向力（年份、月份）和圖片記憶測試（三張圖片延遲回憶）。時間定向力各1分，圖片記憶各1分，總分5分。得分低於4分建議進一步評估。"
    },
    {
        "name": "ICOPE Long Context",
        "prompt": """你是一位ICOPE護理師培訓系統的AI助教。以下是一位護理師與長者的對話紀錄：

護理師：阿嬤好，我是社區護理師小陳，今天來幫您做個健康評估。
長者：好啊，請進請進。
護理師：阿嬤，我先問您幾個問題。您今年幾歲了？
長者：我今年78歲了。
護理師：好的。那您最近有沒有覺得體重有變輕？
長者：有欸，最近半年瘦了大概3公斤。
護理師：那您最近食慾怎麼樣？
長者：還好啦，就是有時候覺得吃東西沒什麼味道。
護理師：了解。那我現在要問您幾個記憶的問題。現在是幾年？
長者：嗯⋯⋯是2025年吧？
護理師：現在是幾月？
長者：呃⋯⋯是不是6月？
護理師：好的，我現在給您看三張圖片，請您記住。（展示圖片）
長者：好的，我看到了。
護理師：那我們過幾分鐘再回來問您記得哪些圖片。

請根據以上對話，分析這位長者在ICOPE六大面向（認知、行動力、營養、視力、聽力、憂鬱）的初步評估結果，並給出建議。""",
        "reference": "根據對話分析：1.認知功能：時間定向力正確（年份、月份皆答對），得2分，需後續確認圖片記憶。2.營養：半年內體重減輕3公斤，食慾下降，味覺改變，營養風險高。3.行動力、視力、聽力、憂鬱：對話中未涉及相關評估。建議：營養方面需轉介營養師評估，認知功能待圖片回憶結果後完整評分。"
    }
]


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using SequenceMatcher"""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1, text2).ratio()


def run_benchmark():
    results = []
    print(f"\n{'='*70}")
    print(f"  MI300X Performance Benchmark - AMD Instinct MI300X (192GB, ROCm 7.1.1)")
    print(f"  Platform TDP: {PLATFORM_TDP_W}W")
    print(f"  Models: {len(MODELS_TO_TEST)} | Test Cases: {len(TEST_CASES)}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    for model in MODELS_TO_TEST:
        print(f"\n🚀 Benchmarking Model: {model}")
        print(f"{'─'*50}")

        for case in TEST_CASES:
            print(f"  ⏳ {case['name']}...", end="", flush=True)

            try:
                start_time = time.time()
                response = ollama.generate(
                    model=model,
                    prompt=case['prompt'],
                    stream=False
                )
                wall_time = time.time() - start_time

                # Extract metrics from Ollama response (durations in nanoseconds)
                eval_count = response.get('eval_count', 0)
                eval_duration_ns = response.get('eval_duration', 1)
                prompt_eval_count = response.get('prompt_eval_count', 0)
                prompt_eval_duration_ns = response.get('prompt_eval_duration', 1)
                total_duration_ns = response.get('total_duration', 0)
                load_duration_ns = response.get('load_duration', 0)

                # Calculate metrics
                decode_tps = (eval_count / eval_duration_ns) * 1e9 if eval_duration_ns > 0 else 0
                prefill_tps = (prompt_eval_count / prompt_eval_duration_ns) * 1e9 if prompt_eval_duration_ns > 0 else 0
                watt_per_token = PLATFORM_TDP_W / decode_tps if decode_tps > 0 else 0

                # Q&A similarity
                output_text = response.get('response', '')
                similarity = calculate_similarity(output_text, case['reference']) if case.get('reference') else None

                benchmark_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "platform": PLATFORM_NAME,
                    "model": model,
                    "test_case": case['name'],
                    "prompt": case['prompt'][:100] + "..." if len(case['prompt']) > 100 else case['prompt'],
                    "metrics": {
                        "context_window_input_tokens": prompt_eval_count,
                        "output_tokens": eval_count,
                        "prefill_tps": round(prefill_tps, 2),
                        "decode_tps": round(decode_tps, 2),
                        "tokens_per_second": round(decode_tps, 2),
                        "watt_per_token": round(watt_per_token, 4),
                        "total_duration_sec": round(total_duration_ns / 1e9, 4),
                        "load_duration_sec": round(load_duration_ns / 1e9, 4),
                        "prompt_eval_duration_sec": round(prompt_eval_duration_ns / 1e9, 4),
                        "eval_duration_sec": round(eval_duration_ns / 1e9, 4),
                        "wall_time_sec": round(wall_time, 4),
                    },
                    "quality": {
                        "similarity_score": round(similarity, 4) if similarity is not None else None,
                        "output_length_chars": len(output_text),
                    },
                    "response_preview": output_text[:200] + "..." if len(output_text) > 200 else output_text
                }

                results.append(benchmark_entry)
                print(f" ✅ Decode:{decode_tps:.1f} tok/s | Prefill:{prefill_tps:.1f} tok/s | Out:{eval_count} tokens | {watt_per_token:.3f} W/tok")

            except Exception as e:
                print(f" ❌ Error: {e}")
                results.append({
                    "timestamp": datetime.now().isoformat(),
                    "platform": PLATFORM_NAME,
                    "model": model,
                    "test_case": case['name'],
                    "error": str(e)
                })

    # Save results
    save_results(results)
    print_summary(results)
    return results


def save_results(data):
    """Save benchmark results to JSON"""
    output_dir = "/root/FY115_LLMCapability_testcase/data/mi300x"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "MI300X-ollama_performance_memory.json")

    try:
        with open(filename, "r") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    history.extend(data)
    with open(filename, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to {filename}")


def print_summary(results):
    """Print performance summary table"""
    print(f"\n{'='*90}")
    print(f"  PERFORMANCE SUMMARY - MI300X ({PLATFORM_TDP_W}W TDP)")
    print(f"{'='*90}")
    print(f"{'Model':<20} {'Test Case':<20} {'Prefill':<12} {'Decode':<12} {'W/Token':<10} {'In/Out':<12} {'Similarity':<10}")
    print(f"{'─'*90}")

    for r in results:
        if 'error' in r:
            print(f"{r['model']:<20} {r['test_case']:<20} {'ERROR':<12}")
            continue
        m = r['metrics']
        q = r['quality']
        sim_str = f"{q['similarity_score']:.3f}" if q['similarity_score'] is not None else "N/A"
        io_str = f"{m['context_window_input_tokens']}/{m['output_tokens']}"
        print(f"{r['model']:<20} {r['test_case']:<20} {m['prefill_tps']:<12.1f} {m['decode_tps']:<12.1f} {m['watt_per_token']:<10.4f} {io_str:<12} {sim_str:<10}")

    # Model averages
    print(f"\n{'─'*90}")
    print(f"{'MODEL AVERAGES':^90}")
    print(f"{'─'*90}")
    print(f"{'Model':<20} {'Avg Decode TPS':<16} {'Avg Prefill TPS':<16} {'Avg W/Token':<12} {'Avg Similarity':<14}")
    print(f"{'─'*90}")

    model_stats = {}
    for r in results:
        if 'error' in r:
            continue
        model = r['model']
        if model not in model_stats:
            model_stats[model] = {'decode': [], 'prefill': [], 'watt': [], 'sim': []}
        model_stats[model]['decode'].append(r['metrics']['decode_tps'])
        model_stats[model]['prefill'].append(r['metrics']['prefill_tps'])
        model_stats[model]['watt'].append(r['metrics']['watt_per_token'])
        if r['quality']['similarity_score'] is not None:
            model_stats[model]['sim'].append(r['quality']['similarity_score'])

    for model, stats in model_stats.items():
        avg_decode = sum(stats['decode']) / len(stats['decode']) if stats['decode'] else 0
        avg_prefill = sum(stats['prefill']) / len(stats['prefill']) if stats['prefill'] else 0
        avg_watt = sum(stats['watt']) / len(stats['watt']) if stats['watt'] else 0
        avg_sim = sum(stats['sim']) / len(stats['sim']) if stats['sim'] else 0
        sim_str = f"{avg_sim:.3f}" if stats['sim'] else "N/A"
        print(f"{model:<20} {avg_decode:<16.2f} {avg_prefill:<16.2f} {avg_watt:<12.4f} {sim_str:<14}")

    print(f"{'='*90}\n")


if __name__ == "__main__":
    run_benchmark()
