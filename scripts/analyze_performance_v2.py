import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import glob
import argparse

# 設置中文字體支援（支援繁體中文）
# 按優先順序嘗試可用的中文字體
matplotlib.rcParams['font.sans-serif'] = [
    'Noto Sans CJK TC',      # 思源黑體繁體
    'WenQuanYi Micro Hei',   # 文泉驛微米黑
    'AR PL UMing TW',        # 文鼎PL明體
    'WenQuanYi Zen Hei',     # 文泉驛正黑
    'SimHei',                # 黑體
    'Microsoft YaHei',       # 微軟雅黑
    'DejaVu Sans'            # 後備字體
]
matplotlib.rcParams['axes.unicode_minus'] = False

# 清除matplotlib的字體快取以確保使用新字體
import matplotlib.font_manager
matplotlib.font_manager._load_fontmanager(try_read_cache=False)

# 定義顏色方案（支援更多平台）
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# 定義平台顯示順序（按性能從高到低）
PLATFORM_ORDER = ['A6000-Ada', 'GB10', 'GTX_1080Ti', 'T4', 'A10']

# 定義各平台的功耗資訊 (TDP in Watts)
PLATFORM_TDP = {
    'A6000-Ada': 300,
    'GB10': 1000,
    'GTX_1080Ti': 250,
    'T4': 70,
    'A10': 150
}

# 定義各平台的FP32算力 (TFLOPS)
PLATFORM_TFLOPS = {
    'A6000-Ada': 91.1,
    'GB10': 5000,
    'GTX_1080Ti': 11.3,
    'T4': 8.1,
    'A10': 31.2
}

def load_data_from_files(json_files):
    """讀取多個平台的性能資料"""
    platform_data = {}
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            if data:
                platform_name = data[0]['platform']
                platform_data[platform_name] = data
                print(f"  ✓ 載入 {platform_name}: {len(data)} 筆測試記錄")
    
    return platform_data

def analyze_data(platform_data):
    """分析並比較多個平台的性能資料"""
    
    # 按照預設順序排列平台，未在列表中的平台放在最後
    available_platforms = list(platform_data.keys())
    platforms = []
    
    # 先添加在PLATFORM_ORDER中的平台
    for p in PLATFORM_ORDER:
        if p in available_platforms:
            platforms.append(p)
    
    # 再添加不在PLATFORM_ORDER中的其他平台
    for p in available_platforms:
        if p not in platforms:
            platforms.append(p)
    
    # 獲取測試案例（使用第一個平台的資料）
    first_platform = platforms[0]
    test_cases = [item['test_case'] for item in platform_data[first_platform]]
    
    results = {
        'platforms': platforms,
        'test_cases': test_cases,
        'tokens_per_second': {},
        'total_duration': {},
        'eval_count': {},
        'load_duration': {},
    }
    
    # 整理每個平台的資料
    for platform in platforms:
        results['tokens_per_second'][platform] = []
        results['total_duration'][platform] = []
        results['eval_count'][platform] = []
        results['load_duration'][platform] = []
        
        for item in platform_data[platform]:
            results['tokens_per_second'][platform].append(item['metrics']['tokens_per_second'])
            results['total_duration'][platform].append(item['metrics']['total_duration_sec'])
            results['eval_count'][platform].append(item['metrics']['eval_count'])
            results['load_duration'][platform].append(item['metrics']['load_duration_sec'])
    
    return results

def plot_comparison(results, model_name='gemma3:4b', output_dir='.'):
    """繪製多平台性能比較圖表"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 建立平台列表字串
    platform_list_str = ', '.join(results['platforms'])
    fig.suptitle(f'Ollama 多平台性能比較分析\n模型: {model_name}\n平台: {platform_list_str}', 
                 fontsize=18, fontweight='bold')
    
    test_cases = results['test_cases']
    platforms = results['platforms']
    x = np.arange(len(test_cases))
    n_platforms = len(platforms)
    width = 0.8 / n_platforms
    
    # 1. Tokens Per Second 比較
    ax1 = fig.add_subplot(gs[0, 0])
    for i, platform in enumerate(platforms):
        tps = results['tokens_per_second'][platform]
        offset = (i - n_platforms/2 + 0.5) * width
        bars = ax1.bar(x + offset, tps, width, label=platform, 
                      color=COLORS[i % len(COLORS)], alpha=0.8)
        
        # 添加數值標籤
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)
    
    ax1.set_ylabel('Tokens Per Second', fontsize=12, fontweight='bold')
    ax1.set_title('推理速度比較 (越高越好)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_cases, rotation=15, ha='right')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 總處理時間比較
    ax2 = fig.add_subplot(gs[0, 1])
    for i, platform in enumerate(platforms):
        dur = results['total_duration'][platform]
        offset = (i - n_platforms/2 + 0.5) * width
        bars = ax2.bar(x + offset, dur, width, label=platform,
                      color=COLORS[i % len(COLORS)], alpha=0.8)
        
        # 添加數值標籤
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8)
    
    ax2.set_ylabel('總處理時間 (秒)', fontsize=12, fontweight='bold')
    ax2.set_title('總處理時間比較 (越低越好)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_cases, rotation=15, ha='right')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. 平均TPS比較（橫條圖）
    ax3 = fig.add_subplot(gs[1, 0])
    avg_tps = [np.mean(results['tokens_per_second'][p]) for p in platforms]
    y_pos = np.arange(len(platforms))
    bars = ax3.barh(y_pos, avg_tps, color=[COLORS[i % len(COLORS)] for i in range(len(platforms))], alpha=0.8)
    
    # 添加數值標籤
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.2f}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(platforms)
    ax3.set_xlabel('平均 Tokens Per Second', fontsize=12, fontweight='bold')
    ax3.set_title('平台平均推理速度排名', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. 相對於最快平台的速度比率
    ax4 = fig.add_subplot(gs[1, 1])
    max_tps = max(avg_tps)
    speedup_ratios = [tps / avg_tps[-1] for tps in avg_tps]  # 相對於最後一個平台
    
    bars = ax4.bar(y_pos, speedup_ratios, color=[COLORS[i % len(COLORS)] for i in range(len(platforms))], alpha=0.8)
    ax4.axhline(y=1, color='r', linestyle='--', linewidth=2, alpha=0.5)
    
    # 添加數值標籤
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax4.set_xticks(y_pos)
    ax4.set_xticklabels(platforms, rotation=15, ha='right')
    ax4.set_ylabel('速度比率', fontsize=12, fontweight='bold')
    ax4.set_title(f'相對於 {platforms[-1]} 的速度比率', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. 效能統計表
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('tight')
    ax5.axis('off')
    
    # 計算統計資料
    table_data = [['平台', '平均 TPS', '最小 TPS', '最大 TPS', '總處理時間(s)', '總Tokens', 'TPS變異度']]
    
    for platform in platforms:
        tps_list = results['tokens_per_second'][platform]
        total_time = sum(results['total_duration'][platform])
        total_tokens = sum(results['eval_count'][platform])
        std_tps = np.std(tps_list)
        
        table_data.append([
            platform,
            f'{np.mean(tps_list):.2f}',
            f'{min(tps_list):.2f}',
            f'{max(tps_list):.2f}',
            f'{total_time:.2f}',
            f'{total_tokens}',
            f'{std_tps:.2f}'
        ])
    
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.12, 0.12, 0.12, 0.15, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # 設置表頭樣式
    for i in range(7):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 設置交替行顏色
    for i in range(1, len(table_data)):
        for j in range(7):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax5.set_title('詳細性能統計摘要', fontsize=14, fontweight='bold', pad=20)
    
    output_path = Path(output_dir) / 'multi_platform_performance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 圖表已儲存: {output_path}")
    
    return avg_tps

def plot_efficiency_comparison(results, avg_tps, model_name='gemma3:4b', output_dir='.'):
    """繪製效能與能耗效率比較圖表"""
    
    platforms = results['platforms']
    
    # 計算各種效率指標
    power_efficiency = {}  # TPS/W
    compute_efficiency = {}  # TPS/TFLOP
    cost_per_million_tokens = {}  # 電費成本
    annual_co2 = {}  # 年度碳排放
    performance_power_ratio = {}  # 性能功耗比
    
    for platform in platforms:
        avg_tps_val = avg_tps[platforms.index(platform)]
        tdp = PLATFORM_TDP.get(platform, 100)
        tflops = PLATFORM_TFLOPS.get(platform, 10)
        
        # 能效比 (TPS/W)
        power_efficiency[platform] = avg_tps_val / tdp
        
        # 算力效率 (TPS/TFLOP)
        compute_efficiency[platform] = avg_tps_val / tflops
        
        # 電費成本 (假設$0.10/kWh，24小時運行)
        daily_kwh = (tdp * 24) / 1000
        tokens_per_day = avg_tps_val * 60 * 60 * 24
        cost_per_million_tokens[platform] = (daily_kwh * 0.10) / (tokens_per_day / 1_000_000)
        
        # 年度碳排放 (kg CO2，假設0.5 kg/kWh)
        annual_kwh = daily_kwh * 365
        annual_co2[platform] = annual_kwh * 0.5
        
        # 性能功耗比 (歸一化)
        performance_power_ratio[platform] = avg_tps_val / tdp * 100
    
    # 創建5個子圖
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    fig.suptitle(f'五大效能能耗評比分析\n模型: {model_name}', 
                 fontsize=20, fontweight='bold')
    
    # 1. 能效比 (TPS/W)
    ax1 = fig.add_subplot(gs[0, 0])
    pe_values = [power_efficiency[p] for p in platforms]
    bars = ax1.barh(platforms, pe_values, color=[COLORS[i % len(COLORS)] for i in range(len(platforms))], alpha=0.8)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}',
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('能效比 (Tokens/秒/瓦)', fontsize=12, fontweight='bold')
    ax1.set_title('① 能耗效率排名 (TPS/W)\n數值越高越好', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. 算力效率 (TPS/TFLOP)
    ax2 = fig.add_subplot(gs[0, 1])
    ce_values = [compute_efficiency[p] for p in platforms]
    bars = ax2.barh(platforms, ce_values, color=[COLORS[i % len(COLORS)] for i in range(len(platforms))], alpha=0.8)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.2f}',
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('算力效率 (TPS/TFLOP)', fontsize=12, fontweight='bold')
    ax2.set_title('② 算力利用效率 (TPS/TFLOP)\n數值越高表示算力轉換效率越好', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. 電費成本 (每百萬Token)
    ax3 = fig.add_subplot(gs[1, 0])
    cost_values = [cost_per_million_tokens[p] for p in platforms]
    bars = ax3.bar(platforms, cost_values, color=[COLORS[i % len(COLORS)] for i in range(len(platforms))], alpha=0.8)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_ylabel('電費成本 (美元)', fontsize=12, fontweight='bold')
    ax3.set_title('③ 每百萬Token電費成本\n數值越低越經濟', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(platforms)))
    ax3.set_xticklabels(platforms, rotation=15, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. 年度碳排放
    ax4 = fig.add_subplot(gs[1, 1])
    co2_values = [annual_co2[p] for p in platforms]
    bars = ax4.bar(platforms, co2_values, color=[COLORS[i % len(COLORS)] for i in range(len(platforms))], alpha=0.8)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax4.set_ylabel('碳排放量 (kg CO₂)', fontsize=12, fontweight='bold')
    ax4.set_title('④ 年度碳排放量 (24/7運行)\n數值越低越環保', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(platforms)))
    ax4.set_xticklabels(platforms, rotation=15, ha='right')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. 綜合性能功耗比雷達圖
    ax5 = fig.add_subplot(gs[2, :], projection='polar')
    
    # 準備雷達圖資料 (歸一化到0-1)
    metrics = {
        '推理速度': [avg_tps[platforms.index(p)] / max(avg_tps) for p in platforms],
        '能效比': [power_efficiency[p] / max(power_efficiency.values()) for p in platforms],
        '算力效率': [compute_efficiency[p] / max(compute_efficiency.values()) for p in platforms],
        '成本優勢': [1 - (cost_per_million_tokens[p] / max(cost_per_million_tokens.values())) for p in platforms],
        '環保指數': [1 - (annual_co2[p] / max(annual_co2.values())) for p in platforms]
    }
    
    # 設置雷達圖
    categories = list(metrics.keys())
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    for i, platform in enumerate(platforms):
        values = [metrics[cat][i] for cat in categories]
        values += values[:1]
        
        ax5.plot(angles, values, 'o-', linewidth=2, label=platform, 
                color=COLORS[i % len(COLORS)])
        ax5.fill(angles, values, alpha=0.15, color=COLORS[i % len(COLORS)])
    
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories, size=12, fontweight='bold')
    ax5.set_ylim(0, 1)
    ax5.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax5.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax5.grid(True)
    ax5.set_title('⑤ 五維綜合評比雷達圖\n(外圈表示最佳，內圈表示最差)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    output_path = Path(output_dir) / 'efficiency_power_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 能耗效率圖表已儲存: {output_path}")
    
    return {
        'power_efficiency': power_efficiency,
        'compute_efficiency': compute_efficiency,
        'cost_per_million_tokens': cost_per_million_tokens,
        'annual_co2': annual_co2
    }

def generate_markdown_report(results, avg_tps, model_name='gemma3:4b', output_dir='.'):
    """生成詳細的Markdown分析報告"""
    
    platforms = results['platforms']
    test_cases = results['test_cases']
    
    # 找出最快和最慢的平台
    fastest_idx = avg_tps.index(max(avg_tps))
    slowest_idx = avg_tps.index(min(avg_tps))
    fastest_platform = platforms[fastest_idx]
    slowest_platform = platforms[slowest_idx]
    speed_diff = avg_tps[fastest_idx] / avg_tps[slowest_idx]
    
    report = f"""# Ollama 多平台性能分析報告
## 模型: {model_name}
## 硬體平台比較: {', '.join(platforms)}

生成位置: {Path(__file__).resolve()}
日期: 2026-01-02

---

## 執行摘要

本報告對比了 **{len(platforms)}個硬體平台** 在運行 `{model_name}` 模型時的推理性能。測試包含三個不同複雜度的場景，從簡短問答到程式碼生成任務。

### 關鍵發現

- 🏆 **最快平台**: **{fastest_platform}** ({avg_tps[fastest_idx]:.2f} tokens/s)
- 🐌 **最慢平台**: **{slowest_platform}** ({avg_tps[slowest_idx]:.2f} tokens/s)
- ⚡ **性能差距**: 最快平台比最慢平台快 **{speed_diff:.2f}x**
- 📊 **平台總數**: {len(platforms)} 個硬體平台

---

## 平台性能排名

### 平均吞吐量排名 (Tokens Per Second)

"""
    
    # 創建排名列表
    ranked_platforms = sorted(zip(platforms, avg_tps), key=lambda x: x[1], reverse=True)
    
    report += "| 排名 | 平台 | 平均 TPS | 相對最快平台 | 相對最慢平台 |\n"
    report += "|------|------|----------|--------------|-------------|\n"
    
    for rank, (platform, tps) in enumerate(ranked_platforms, 1):
        rel_to_fastest = tps / avg_tps[fastest_idx]
        rel_to_slowest = tps / avg_tps[slowest_idx]
        
        if rank == 1:
            emoji = "🥇"
        elif rank == 2:
            emoji = "🥈"
        elif rank == 3:
            emoji = "🥉"
        else:
            emoji = f"  {rank}"
        
        report += f"| {emoji} | **{platform}** | {tps:.2f} | {rel_to_fastest:.2f}x | {rel_to_slowest:.2f}x |\n"
    
    report += f"""
---

## 詳細性能指標

### 1. Tokens Per Second (TPS) 各測試案例比較

| 測試案例 | {' | '.join(platforms)} |
|----------|{'|'.join(['----------' for _ in platforms])}|
"""
    
    for i, test_case in enumerate(test_cases):
        row = f"| {test_case} |"
        for platform in platforms:
            tps = results['tokens_per_second'][platform][i]
            row += f" {tps:.2f} |"
        report += row + "\n"
    
    # 添加平均值行
    report += f"| **平均** |"
    for platform in platforms:
        avg = np.mean(results['tokens_per_second'][platform])
        report += f" **{avg:.2f}** |"
    report += "\n\n"
    
    report += """### 2. 總處理時間比較 (秒)

"""
    
    report += f"| 測試案例 | {' | '.join(platforms)} |\n"
    report += f"|----------|{'|'.join(['----------' for _ in platforms])}|\n"
    
    for i, test_case in enumerate(test_cases):
        row = f"| {test_case} |"
        for platform in platforms:
            dur = results['total_duration'][platform][i]
            row += f" {dur:.3f} |"
        report += row + "\n"
    
    # 添加總計行
    report += f"| **總計** |"
    for platform in platforms:
        total = sum(results['total_duration'][platform])
        report += f" **{total:.3f}** |"
    report += "\n\n"
    
    report += """### 3. 生成的Token數量

"""
    
    report += f"| 測試案例 | {' | '.join(platforms)} |\n"
    report += f"|----------|{'|'.join(['----------' for _ in platforms])}|\n"
    
    for i, test_case in enumerate(test_cases):
        row = f"| {test_case} |"
        for platform in platforms:
            tokens = results['eval_count'][platform][i]
            row += f" {tokens} |"
        report += row + "\n"
    
    # 添加總計行
    report += f"| **總計** |"
    for platform in platforms:
        total = sum(results['eval_count'][platform])
        report += f" **{total}** |"
    report += "\n\n"
    
    report += """
---

## 分析與洞察

### 性能特性分析

"""
    
    # 對每個平台進行分析
    for i, platform in enumerate(platforms):
        tps_list = results['tokens_per_second'][platform]
        avg = np.mean(tps_list)
        std = np.std(tps_list)
        total_time = sum(results['total_duration'][platform])
        
        rank = [p for p, _ in ranked_platforms].index(platform) + 1
        
        report += f"""
#### {rank}. {platform}
- **平均TPS**: {avg:.2f}
- **TPS範圍**: {min(tps_list):.2f} - {max(tps_list):.2f}
- **TPS標準差**: {std:.2f} ({"穩定" if std < 5 else "變化較大"})
- **總處理時間**: {total_time:.2f}秒
- **相對{fastest_platform}**: {(avg/avg_tps[fastest_idx]):.2f}x {"" if platform == fastest_platform else f"({((1-avg/avg_tps[fastest_idx])*100):.1f}% 較慢)"}
"""
    
    report += """
---

## 任務特定分析

"""
    
    for i, test_case in enumerate(test_cases):
        report += f"\n### {test_case}\n\n"
        
        # 找出該任務最快和最慢的平台
        task_tps = [results['tokens_per_second'][p][i] for p in platforms]
        task_fastest_idx = task_tps.index(max(task_tps))
        task_slowest_idx = task_tps.index(min(task_tps))
        
        report += f"- **最快**: {platforms[task_fastest_idx]} ({task_tps[task_fastest_idx]:.2f} TPS)\n"
        report += f"- **最慢**: {platforms[task_slowest_idx]} ({task_tps[task_slowest_idx]:.2f} TPS)\n"
        report += f"- **性能差距**: {task_tps[task_fastest_idx]/task_tps[task_slowest_idx]:.2f}x\n"
        
        report += "\n**各平台表現**:\n"
        for platform in platforms:
            tps = results['tokens_per_second'][platform][i]
            tokens = results['eval_count'][platform][i]
            duration = results['total_duration'][platform][i]
            report += f"- {platform}: {tps:.2f} TPS, {tokens} tokens, {duration:.2f}s\n"
    
    report += """
---

## 建議

### 生產環境部署：

"""
    
    # 推薦前3名
    top3 = ranked_platforms[:3]
    for rank, (platform, tps) in enumerate(top3, 1):
        if rank == 1:
            report += f"- ✅ **強烈推薦 {platform}**: 最佳性能 ({tps:.2f} TPS)，適合對延遲敏感的即時應用\n"
        elif rank == 2:
            rel = tps / top3[0][1]
            report += f"- 👍 **推薦 {platform}**: 優秀性能 ({tps:.2f} TPS, {rel:.2%} of {top3[0][0]})，性價比選擇\n"
        else:
            rel = tps / top3[0][1]
            report += f"- 💡 **可考慮 {platform}**: 良好性能 ({tps:.2f} TPS, {rel:.2%} of {top3[0][0]})，適合一般應用\n"
    
    report += """
### 開發與測試：

"""
    
    # 提及後面的平台
    if len(platforms) > 3:
        report += "以下平台適合開發測試環境：\n"
        for platform, tps in ranked_platforms[3:]:
            report += f"- {platform}: {tps:.2f} TPS - 適合開發、除錯、低流量測試\n"
    
    report += f"""
### 成本效益分析：

- **{fastest_platform}** 在相同時間內可處理 **{speed_diff:.2f}x** 於 **{slowest_platform}** 的請求
- 對於高流量生產環境，選擇高性能平台可顯著減少所需伺服器數量
- 需要根據實際流量需求、預算和延遲要求選擇合適的平台

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

### 測試平台
"""
    
    for i, platform in enumerate(platforms, 1):
        report += f"{i}. **{platform}**\n"
    
    report += """
---

## 視覺化

![多平台性能比較圖表](multi_platform_performance_comparison.png)

圖表包含：
1. **左上**: 各測試案例的推理速度比較 (TPS)
2. **右上**: 各測試案例的總處理時間
3. **左中**: 平台平均推理速度排名
4. **右中**: 相對速度比率
5. **下方**: 詳細性能統計摘要表

---

## 結論

"""
    
    report += f"""
本次測試對比了 **{len(platforms)}個硬體平台** 在 gemma3:4b 模型推理任務上的性能表現。

**關鍵結論**:
- 🏆 **{fastest_platform}** 展現最佳性能，平均 {avg_tps[fastest_idx]:.2f} TPS
- 📊 性能差距範圍從 {min(avg_tps):.2f} 到 {max(avg_tps):.2f} TPS
- ⚡ 最快與最慢平台相差 **{speed_diff:.2f}x**

**選擇建議**:
- **即時應用、高吞吐量**: 選擇 {fastest_platform}
- **性價比平衡**: 考慮 {ranked_platforms[1][0] if len(ranked_platforms) > 1 else fastest_platform}
- **開發測試**: 可使用性能較低的平台以降低成本

根據實際應用場景、預算限制和性能需求，選擇最適合的硬體平台。

---

*報告由 analyze_performance_v2.py 自動生成*
"""
    
    output_path = Path(output_dir) / 'multi_platform_performance_report.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 報告已儲存: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='分析多平台Ollama性能資料')
    parser.add_argument('--files', nargs='+', help='指定JSON檔案路徑（可多個）')
    parser.add_argument('--pattern', default='*-ollama_performance_memory.json',
                       help='JSON檔案的匹配模式（預設: *-ollama_performance_memory.json）')
    parser.add_argument('--model', default='gemma3:4b',
                       help='模型名稱（預設: gemma3:4b）')
    parser.add_argument('--output-dir', default='.',
                       help='輸出目錄（預設: 當前目錄）')
    
    args = parser.parse_args()
    
    # 如果指定了檔案，使用指定的檔案；否則使用模式匹配
    if args.files:
        json_files = args.files
    else:
        json_files = sorted(glob.glob(args.pattern))
    
    if not json_files:
        print("❌ 錯誤：找不到符合條件的JSON檔案")
        print(f"   搜尋模式: {args.pattern}")
        return
    
    print(f"🔍 找到 {len(json_files)} 個JSON檔案")
    print(f"📂 準備載入以下檔案:")
    for f in json_files:
        print(f"   - {f}")
    print()
    
    print("🔍 載入性能資料...")
    platform_data = load_data_from_files(json_files)
    
    if len(platform_data) < 2:
        print("❌ 錯誤：至少需要2個平台的資料才能進行比較")
        return
    
    print(f"\n📊 分析 {len(platform_data)} 個平台的性能指標...")
    results = analyze_data(platform_data)
    
    print("📈 生成比較圖表...")
    avg_tps = plot_comparison(results, args.model, args.output_dir)
    
    print("⚡ 生成能耗效率圖表...")
    efficiency_data = plot_efficiency_comparison(results, avg_tps, args.model, args.output_dir)
    
    print("📝 生成markdown報告...")
    generate_markdown_report(results, avg_tps, args.model, args.output_dir)
    
    print("\n" + "="*70)
    print("✅ 分析完成！")
    print("="*70)
    output_dir_path = Path(args.output_dir)
    print(f"📊 性能圖表: {output_dir_path / 'multi_platform_performance_comparison.png'}")
    print(f"⚡ 能耗圖表: {output_dir_path / 'efficiency_power_comparison.png'}")
    print(f"📝 詳細報告: {output_dir_path / 'multi_platform_performance_report.md'}")
    print(f"\n🏆 最佳平台: {results['platforms'][avg_tps.index(max(avg_tps))]} ({max(avg_tps):.2f} TPS)")
    print(f"📉 最慢平台: {results['platforms'][avg_tps.index(min(avg_tps))]} ({min(avg_tps):.2f} TPS)")
    print(f"⚡ 性能差距: {max(avg_tps)/min(avg_tps):.2f}x")
    
    # 顯示能效資訊
    best_efficiency = max(efficiency_data['power_efficiency'].items(), key=lambda x: x[1])
    print(f"\n🌟 最佳能效: {best_efficiency[0]} ({best_efficiency[1]:.3f} TPS/W)")
    
    lowest_cost = min(efficiency_data['cost_per_million_tokens'].items(), key=lambda x: x[1])
    print(f"💰 最低成本: {lowest_cost[0]} (每百萬Token ${lowest_cost[1]:.3f})")
    
    lowest_co2 = min(efficiency_data['annual_co2'].items(), key=lambda x: x[1])
    print(f"🌱 最環保: {lowest_co2[0]} (年排放 {lowest_co2[1]:.0f} kg CO₂)")
    print("="*70)

if __name__ == "__main__":
    main()
