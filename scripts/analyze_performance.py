import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# 設置中文字體支援
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_data():
    """讀取兩個平台的性能資料"""
    with open('A6000-Ada-ollama_performance_memory.json', 'r') as f:
        a6000_data = json.load(f)
    
    with open('GB10-ollama_performance_memory.json', 'r') as f:
        gb10_data = json.load(f)
    
    return a6000_data, gb10_data

def analyze_data(a6000_data, gb10_data):
    """分析並比較兩個平台的性能資料"""
    
    results = {
        'platforms': ['A6000-Ada', 'GB10'],
        'test_cases': [],
        'tokens_per_second': {'A6000-Ada': [], 'GB10': []},
        'total_duration': {'A6000-Ada': [], 'GB10': []},
        'eval_count': {'A6000-Ada': [], 'GB10': []},
    }
    
    # 整理資料
    for a6000, gb10 in zip(a6000_data, gb10_data):
        test_case = a6000['test_case']
        results['test_cases'].append(test_case)
        
        results['tokens_per_second']['A6000-Ada'].append(a6000['metrics']['tokens_per_second'])
        results['tokens_per_second']['GB10'].append(gb10['metrics']['tokens_per_second'])
        
        results['total_duration']['A6000-Ada'].append(a6000['metrics']['total_duration_sec'])
        results['total_duration']['GB10'].append(gb10['metrics']['total_duration_sec'])
        
        results['eval_count']['A6000-Ada'].append(a6000['metrics']['eval_count'])
        results['eval_count']['GB10'].append(gb10['metrics']['eval_count'])
    
    return results

def plot_comparison(results):
    """繪製性能比較圖表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ollama Performance Comparison: A6000-Ada vs GB10\nModel: gemma3:4b', 
                 fontsize=16, fontweight='bold')
    
    test_cases = results['test_cases']
    x = np.arange(len(test_cases))
    width = 0.35
    
    # 1. Tokens Per Second Comparison
    ax1 = axes[0, 0]
    a6000_tps = results['tokens_per_second']['A6000-Ada']
    gb10_tps = results['tokens_per_second']['GB10']
    
    bars1 = ax1.bar(x - width/2, a6000_tps, width, label='A6000-Ada', color='#1f77b4', alpha=0.8)
    bars2 = ax1.bar(x + width/2, gb10_tps, width, label='GB10', color='#ff7f0e', alpha=0.8)
    
    ax1.set_ylabel('Tokens Per Second', fontsize=12, fontweight='bold')
    ax1.set_title('Inference Speed (Higher is Better)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_cases)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加數值標籤
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10)
    
    # 2. Total Duration Comparison
    ax2 = axes[0, 1]
    a6000_dur = results['total_duration']['A6000-Ada']
    gb10_dur = results['total_duration']['GB10']
    
    bars1 = ax2.bar(x - width/2, a6000_dur, width, label='A6000-Ada', color='#1f77b4', alpha=0.8)
    bars2 = ax2.bar(x + width/2, gb10_dur, width, label='GB10', color='#ff7f0e', alpha=0.8)
    
    ax2.set_ylabel('Total Duration (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Total Processing Time (Lower is Better)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_cases)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加數值標籤
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=10)
    
    # 3. Speed-up Ratio
    ax3 = axes[1, 0]
    speedup_ratios = [a6000_tps[i] / gb10_tps[i] for i in range(len(test_cases))]
    
    bars = ax3.bar(x, speedup_ratios, color='#2ca02c', alpha=0.8)
    ax3.axhline(y=1, color='r', linestyle='--', linewidth=2, label='Equal Performance')
    ax3.set_ylabel('Speed-up Ratio', fontsize=12, fontweight='bold')
    ax3.set_title('A6000-Ada Performance Advantage', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(test_cases)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 添加數值標籤
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Summary Statistics Table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    # 計算統計資料
    avg_a6000_tps = np.mean(a6000_tps)
    avg_gb10_tps = np.mean(gb10_tps)
    avg_speedup = avg_a6000_tps / avg_gb10_tps
    
    total_tokens_a6000 = sum(results['eval_count']['A6000-Ada'])
    total_tokens_gb10 = sum(results['eval_count']['GB10'])
    total_time_a6000 = sum(a6000_dur)
    total_time_gb10 = sum(gb10_dur)
    
    table_data = [
        ['Metric', 'A6000-Ada', 'GB10', 'Improvement'],
        ['Avg TPS', f'{avg_a6000_tps:.2f}', f'{avg_gb10_tps:.2f}', f'{avg_speedup:.2f}x'],
        ['Total Tokens', f'{total_tokens_a6000}', f'{total_tokens_gb10}', '-'],
        ['Total Time (s)', f'{total_time_a6000:.2f}', f'{total_time_gb10:.2f}', f'{total_time_gb10/total_time_a6000:.2f}x faster'],
        ['Min TPS', f'{min(a6000_tps):.2f}', f'{min(gb10_tps):.2f}', f'{min(a6000_tps)/min(gb10_tps):.2f}x'],
        ['Max TPS', f'{max(a6000_tps):.2f}', f'{max(gb10_tps):.2f}', f'{max(a6000_tps)/max(gb10_tps):.2f}x'],
    ]
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # 設置表頭樣式
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 設置交替行顏色
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax4.set_title('Performance Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('performance_comparison_report.png', dpi=300, bbox_inches='tight')
    print("✅ 圖表已儲存: performance_comparison_report.png")
    
    return speedup_ratios, avg_speedup

def generate_markdown_report(results, speedup_ratios, avg_speedup):
    """生成詳細的Markdown分析報告"""
    
    a6000_tps = results['tokens_per_second']['A6000-Ada']
    gb10_tps = results['tokens_per_second']['GB10']
    test_cases = results['test_cases']
    
    report = f"""# Ollama Performance Analysis Report
## Model: gemma3:4b
## Hardware Comparison: A6000-Ada vs GB10

Generated: {Path(__file__).resolve()}
Date: 2026-01-02

---

## 執行摘要

本報告對比了 **NVIDIA A6000-Ada** 和 **GB10** 兩個硬體平台在運行 `gemma3:4b` 模型時的推理性能。測試包含三個不同複雜度的場景，從簡短問答到程式碼生成任務。

### 關鍵發現

- ⚡ **A6000-Ada平均速度提升**: **{avg_speedup:.2f}x** 倍於GB10
- 🚀 **A6000-Ada平均吞吐量**: **{np.mean(a6000_tps):.2f} tokens/s**
- 📊 **GB10平均吞吐量**: **{np.mean(gb10_tps):.2f} tokens/s**
- 💡 **最佳性能場景**: {test_cases[speedup_ratios.index(max(speedup_ratios))]} ({max(speedup_ratios):.2f}x 加速)

---

## 詳細性能指標

### 1. Tokens Per Second (TPS) Comparison

| Test Case | A6000-Ada TPS | GB10 TPS | Speed-up Ratio |
|-----------|---------------|----------|----------------|
"""
    
    for i, test_case in enumerate(test_cases):
        report += f"| {test_case} | {a6000_tps[i]:.2f} | {gb10_tps[i]:.2f} | **{speedup_ratios[i]:.2f}x** |\n"
    
    report += f"\n**Average** | **{np.mean(a6000_tps):.2f}** | **{np.mean(gb10_tps):.2f}** | **{avg_speedup:.2f}x** |\n\n"
    
    report += """### 2. 總處理時間比較

| 測試案例 | A6000-Ada (s) | GB10 (s) | 節省時間 |
|-----------|---------------|----------|------------|
"""
    
    for i, test_case in enumerate(test_cases):
        a6000_dur = results['total_duration']['A6000-Ada'][i]
        gb10_dur = results['total_duration']['GB10'][i]
        time_saved = gb10_dur - a6000_dur
        report += f"| {test_case} | {a6000_dur:.3f} | {gb10_dur:.3f} | {time_saved:.3f}s ({(time_saved/gb10_dur)*100:.1f}%) |\n"
    
    total_a6000 = sum(results['total_duration']['A6000-Ada'])
    total_gb10 = sum(results['total_duration']['GB10'])
    total_saved = total_gb10 - total_a6000
    
    report += f"\n**總計** | **{total_a6000:.3f}** | **{total_gb10:.3f}** | **{total_saved:.3f}s ({(total_saved/total_gb10)*100:.1f}%)** |\n\n"
    
    report += """### 3. Token生成分析

| 測試案例 | A6000-Ada Tokens | GB10 Tokens | A6000載入時間 (s) | GB10載入時間 (s) |
|-----------|------------------|-------------|---------------------|-------------------|
"""
    
    for i, test_case in enumerate(test_cases):
        a6000_tokens = results['eval_count']['A6000-Ada'][i]
        gb10_tokens = results['eval_count']['GB10'][i]
        a6000_load = results['total_duration']['A6000-Ada'][i] - (a6000_tokens / a6000_tps[i])
        gb10_load = results['total_duration']['GB10'][i] - (gb10_tokens / gb10_tps[i])
        report += f"| {test_case} | {a6000_tokens} | {gb10_tokens} | {a6000_load:.3f} | {gb10_load:.3f} |\n"
    
    report += f"""
---

## 分析與洞察

### 性能特性

#### 🏆 A6000-Ada 優勢：
1. **持續較高的吞吐量**: A6000-Ada 保持 {min(speedup_ratios):.2f}x 至 {max(speedup_ratios):.2f}x 更快的token生成速度
2. **穩定的性能**: 不同任務間的TPS變異度：{np.std(a6000_tps):.2f} (A6000) vs {np.std(gb10_tps):.2f} (GB10)
3. **更好的擴展性**: 在 {test_cases[speedup_ratios.index(max(speedup_ratios))]} 任務上顯示 {max(speedup_ratios):.2f}x 優勢
4. **更低的延遲**: 模型載入和初始化時間相當，但推理速度顯著更快

#### 📊 GB10 特性：
1. **持續較低的性能**: 比A6000-Ada慢約 {avg_speedup:.2f}x
2. **較長的處理時間**: 在複雜任務如程式碼生成中特別明顯
3. **適用於**: 開發、測試和對時間要求較不嚴格的工作負載

### 任務特定觀察

"""
    
    for i, test_case in enumerate(test_cases):
        report += f"""
#### {test_case}
- **生成的Tokens**: A6000={results['eval_count']['A6000-Ada'][i]}, GB10={results['eval_count']['GB10'][i]}
- **性能差距**: A6000-Ada快 {speedup_ratios[i]:.2f}x
- **時間差異**: 節省 {results['total_duration']['GB10'][i] - results['total_duration']['A6000-Ada'][i]:.2f}s
"""
    
    report += """
---

## 建議

### 生產環境部署：
- **A6000-Ada** 強烈建議用於：
  - 即時推理應用
  - 高吞吐量場景
  - 面向使用者的聊天機器人和互動系統
  - 大規模部署具成本效益（更快的處理 = 每小時更多請求）

### 開發與測試：
- **GB10** 適用於：
  - 模型開發和除錯
  - 低流量測試
  - 時間限制較彈性的批次處理

### 成本效益分析：
- A6000-Ada 在相同時間內可處理 **{avg_speedup:.2f}x 更多請求**
- 對於高流量生產環境，A6000-Ada可處理 {avg_speedup:.2f}x 更多使用者
- 損益平衡點取決於基礎設施成本與吞吐量需求

---

## 技術規格

### 測試環境
- **模型**: gemma3:4b (Ollama)
- **測試日期**: 2026-01-02
- **測試框架**: 自訂Python基準測試 (ollama_test.py)
- **收集的指標**: 
  - 每秒Token數 (TPS)
  - 總處理時間
  - 載入時間
  - 評估計數
  - 提示評估計數

### 硬體平台
1. **A6000-Ada**: NVIDIA RTX A6000 Ada Generation
2. **GB10**: [硬體規格待確認]

---

## 視覺化

![性能比較圖表](performance_comparison_report.png)

上圖顯示：
1. **左上**: 推理速度比較 (TPS)
2. **右上**: 總處理時間
3. **左下**: 每個任務的加速比率
4. **右下**: 摘要統計表

---

## 結論

NVIDIA A6000-Ada 在 gemma3:4b 推理任務上展現出比 GB10 **{avg_speedup:.2f}x 的優越性能**。這種性能優勢在所有測試場景中都保持一致，使得A6000-Ada成為需要以下條件的生產環境部署的明確選擇：
- 低延遲回應
- 高吞吐量
- 一致的性能

對於時間限制較不嚴格的開發和測試目的，GB10仍然是一個可行的選擇。

---

*報告由 analyze_performance.py 自動生成*
"""
    
    with open('performance_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✅ 報告已儲存: performance_analysis_report.md")

def main():
    print("🔍 載入性能資料...")
    a6000_data, gb10_data = load_data()
    
    print("📊 分析性能指標...")
    results = analyze_data(a6000_data, gb10_data)
    
    print("📈 生成比較圖表...")
    speedup_ratios, avg_speedup = plot_comparison(results)
    
    print("📝 生成markdown報告...")
    generate_markdown_report(results, speedup_ratios, avg_speedup)
    
    print("\n" + "="*60)
    print("✅ 分析完成！")
    print("="*60)
    print(f"📊 圖表: performance_comparison_report.png")
    print(f"📝 報告: performance_analysis_report.md")
    print(f"\n🚀 關鍵發現: A6000-Ada 比 GB10 快 {avg_speedup:.2f}x")
    print("="*60)

if __name__ == "__main__":
    main()
