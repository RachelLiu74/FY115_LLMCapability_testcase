#!/usr/bin/env python3
"""
MI300X Performance Benchmark - Chart Generator
Generates performance comparison charts from benchmark data
"""

import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

# Font setup - directly register the CJK font file
_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(_FONT_PATH)
_font_prop = fm.FontProperties(fname=_FONT_PATH)
_font_name = _font_prop.get_name()

matplotlib.rcParams['font.sans-serif'] = [_font_name, 'DejaVu Sans']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

# Colors
COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#00BCD4', '#FF5722']

OUTPUT_DIR = Path("/root/FY115_LLMCapability_testcase/reports")

def load_data():
    data_file = Path("/root/FY115_LLMCapability_testcase/data/mi300x/MI300X-ollama_performance_memory.json")
    with open(data_file, 'r') as f:
        return json.load(f)

def organize_data(raw_data):
    """Organize raw data by model"""
    models = {}
    for entry in raw_data:
        if 'error' in entry:
            continue
        model = entry['model']
        if model not in models:
            models[model] = []
        models[model].append(entry)
    return models

def plot_decode_speed(models_data):
    """Chart 1: Decode Speed (tok/s) comparison"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    model_names = list(models_data.keys())
    test_cases = ['Short QA', 'Creative Writing', 'Code Gen', 'ICOPE Nursing QA', 'ICOPE Long Context']
    
    x = np.arange(len(model_names))
    width = 0.15
    
    for i, tc in enumerate(test_cases):
        values = []
        for model in model_names:
            entries = models_data[model]
            val = next((e['metrics']['decode_tps'] for e in entries if e['test_case'] == tc), 0)
            values.append(val)
        bars = ax.bar(x + i * width, values, width, label=tc, color=COLORS[i], alpha=0.85)
    
    ax.set_xlabel('模型', fontsize=12)
    ax.set_ylabel('Decode 速度 (tokens/s)', fontsize=12)
    ax.set_title('MI300X Decode 速度比較 (越高越好)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(250, ax.get_ylim()[1] * 1.1))
    
    plt.tight_layout()
    path = OUTPUT_DIR / "MI300X_decode_speed.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")

def plot_prefill_speed(models_data):
    """Chart 2: Prefill Speed comparison"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    model_names = list(models_data.keys())
    test_cases = ['Short QA', 'Creative Writing', 'Code Gen', 'ICOPE Nursing QA', 'ICOPE Long Context']
    
    x = np.arange(len(model_names))
    width = 0.15
    
    for i, tc in enumerate(test_cases):
        values = []
        for model in model_names:
            entries = models_data[model]
            val = next((e['metrics']['prefill_tps'] for e in entries if e['test_case'] == tc), 0)
            values.append(val)
        ax.bar(x + i * width, values, width, label=tc, color=COLORS[i], alpha=0.85)
    
    ax.set_xlabel('模型', fontsize=12)
    ax.set_ylabel('Prefill 速度 (tokens/s)', fontsize=12)
    ax.set_title('MI300X Prefill 速度比較 (越高越好)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = OUTPUT_DIR / "MI300X_prefill_speed.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")

def plot_model_avg_comparison(models_data):
    """Chart 3: Model average performance bar chart"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    model_names = list(models_data.keys())
    
    # Calculate averages
    avg_decode = []
    avg_prefill = []
    avg_watt = []
    avg_sim = []
    
    for model in model_names:
        entries = models_data[model]
        decode_vals = [e['metrics']['decode_tps'] for e in entries]
        prefill_vals = [e['metrics']['prefill_tps'] for e in entries]
        watt_vals = [e['metrics']['watt_per_token'] for e in entries]
        sim_vals = [e['quality']['similarity_score'] for e in entries if e['quality']['similarity_score'] is not None]
        
        avg_decode.append(np.mean(decode_vals))
        avg_prefill.append(np.mean(prefill_vals))
        avg_watt.append(np.mean(watt_vals))
        avg_sim.append(np.mean(sim_vals) if sim_vals else 0)
    
    x = np.arange(len(model_names))
    
    # Subplot 1: Decode TPS
    ax = axes[0, 0]
    bars = ax.barh(x, avg_decode, color=COLORS[:len(model_names)], alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(model_names)
    ax.set_xlabel('tok/s')
    ax.set_title('平均 Decode 速度', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, avg_decode):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}', va='center', fontsize=9)
    
    # Subplot 2: Prefill TPS
    ax = axes[0, 1]
    bars = ax.barh(x, avg_prefill, color=COLORS[:len(model_names)], alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(model_names)
    ax.set_xlabel('tok/s')
    ax.set_title('平均 Prefill 速度', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, avg_prefill):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, f'{val:.0f}', va='center', fontsize=9)
    
    # Subplot 3: Watt per Token (lower is better)
    ax = axes[1, 0]
    bars = ax.barh(x, avg_watt, color=COLORS[:len(model_names)], alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(model_names)
    ax.set_xlabel('W/Token')
    ax.set_title('能耗效率 W/Token (越低越好)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, avg_watt):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center', fontsize=9)
    
    # Subplot 4: Similarity Score
    ax = axes[1, 1]
    bars = ax.barh(x, avg_sim, color=COLORS[:len(model_names)], alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(model_names)
    ax.set_xlabel('Similarity')
    ax.set_title('Q&A 回答品質 (相似度)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max(avg_sim) * 1.3)
    for bar, val in zip(bars, avg_sim):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
    
    fig.suptitle('MI300X LLM 效能總覽 (AMD Instinct MI300X, 192GB, ROCm 7.1.1)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUTPUT_DIR / "MI300X_performance_overview.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")

def plot_efficiency_radar(models_data):
    """Chart 4: Radar chart - multi-dimensional comparison"""
    model_names = list(models_data.keys())
    
    # Calculate normalized metrics
    metrics = {'Decode速度': [], 'Prefill速度': [], '能效(1/W)': [], '回答品質': [], '輸出量': []}
    
    for model in model_names:
        entries = models_data[model]
        decode_vals = [e['metrics']['decode_tps'] for e in entries]
        prefill_vals = [e['metrics']['prefill_tps'] for e in entries]
        watt_vals = [e['metrics']['watt_per_token'] for e in entries]
        sim_vals = [e['quality']['similarity_score'] for e in entries if e['quality']['similarity_score'] is not None]
        output_vals = [e['metrics']['output_tokens'] for e in entries]
        
        metrics['Decode速度'].append(np.mean(decode_vals))
        metrics['Prefill速度'].append(np.mean(prefill_vals))
        metrics['能效(1/W)'].append(1.0 / np.mean(watt_vals))  # Invert: lower W/tok = better
        metrics['回答品質'].append(np.mean(sim_vals) if sim_vals else 0)
        metrics['輸出量'].append(np.mean(output_vals))
    
    # Normalize to 0-1
    categories = list(metrics.keys())
    normalized = {}
    for cat in categories:
        vals = metrics[cat]
        max_val = max(vals) if max(vals) > 0 else 1
        normalized[cat] = [v / max_val for v in vals]
    
    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for i, model in enumerate(model_names):
        values = [normalized[cat][i] for cat in categories]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=COLORS[i], alpha=0.8)
        ax.fill(angles, values, alpha=0.1, color=COLORS[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title('MI300X 模型五維綜合評比', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = OUTPUT_DIR / "MI300X_radar_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")

def plot_watt_efficiency(models_data):
    """Chart 5: Energy efficiency and cost analysis"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    model_names = list(models_data.keys())
    TDP = 750  # MI300X TDP
    
    avg_decode = []
    for model in model_names:
        entries = models_data[model]
        avg_decode.append(np.mean([e['metrics']['decode_tps'] for e in entries]))
    
    # TPS/W efficiency
    tps_per_watt = [d / TDP for d in avg_decode]
    
    ax = axes[0]
    bars = ax.barh(model_names, tps_per_watt, color=COLORS[:len(model_names)], alpha=0.85)
    ax.set_xlabel('TPS/W (越高越好)')
    ax.set_title('能效比 (TPS per Watt)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, tps_per_watt):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
    
    # Cost per million tokens (electricity at $0.10/kWh)
    cost_per_million = []
    for d in avg_decode:
        daily_kwh = (TDP * 24) / 1000
        daily_tokens = d * 3600 * 24
        cost = (daily_kwh * 0.10) / (daily_tokens / 1_000_000)
        cost_per_million.append(cost)
    
    ax = axes[1]
    bars = ax.barh(model_names, cost_per_million, color=COLORS[:len(model_names)], alpha=0.85)
    ax.set_xlabel('USD (越低越好)')
    ax.set_title('每百萬 Token 電費成本', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, cost_per_million):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, f'${val:.4f}', va='center', fontsize=9)
    
    # Annual CO2 emissions
    annual_co2 = []
    for d in avg_decode:
        annual_kwh = (TDP * 24 * 365) / 1000
        co2 = annual_kwh * 0.5  # 0.5 kg CO2/kWh
        annual_co2.append(co2)
    
    ax = axes[2]
    # Same for all models since TDP is fixed
    bars = ax.barh(model_names, [annual_co2[0]] * len(model_names), color='#FF5722', alpha=0.6)
    ax.set_xlabel('kg CO2/年')
    ax.set_title(f'年度碳排放 (TDP={TDP}W, 24/7)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.text(0.5, 0.95, f'固定 TDP: {annual_co2[0]:.0f} kg CO2/年', transform=ax.transAxes, 
            ha='center', va='top', fontsize=11, color='red', fontweight='bold')
    
    fig.suptitle('MI300X 能耗與成本分析 (TDP: 750W)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUTPUT_DIR / "MI300X_efficiency_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")

def plot_io_tokens(models_data):
    """Chart 6: Input/Output token distribution"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    model_names = list(models_data.keys())
    test_cases = ['Short QA', 'Creative Writing', 'Code Gen', 'ICOPE Nursing QA', 'ICOPE Long Context']
    
    x = np.arange(len(test_cases))
    width = 0.12
    
    for i, model in enumerate(model_names):
        entries = models_data[model]
        output_tokens = []
        for tc in test_cases:
            val = next((e['metrics']['output_tokens'] for e in entries if e['test_case'] == tc), 0)
            output_tokens.append(val)
        ax.bar(x + i * width, output_tokens, width, label=model, color=COLORS[i], alpha=0.85)
    
    ax.set_xlabel('測試案例', fontsize=12)
    ax.set_ylabel('Output Tokens', fontsize=12)
    ax.set_title('MI300X 各模型輸出 Token 數量比較', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(test_cases, rotation=10, ha='right')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = OUTPUT_DIR / "MI300X_output_tokens.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")


def main():
    print("\n📊 MI300X Performance Chart Generator")
    print("=" * 50)
    
    raw_data = load_data()
    models_data = organize_data(raw_data)
    
    print(f"\n  載入 {len(raw_data)} 筆測試資料 ({len(models_data)} 個模型)")
    print(f"\n  生成圖表中...\n")
    
    plot_decode_speed(models_data)
    plot_prefill_speed(models_data)
    plot_model_avg_comparison(models_data)
    plot_efficiency_radar(models_data)
    plot_watt_efficiency(models_data)
    plot_io_tokens(models_data)
    
    print(f"\n{'='*50}")
    print(f"  ✅ 所有圖表已輸出至: {OUTPUT_DIR}/")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
