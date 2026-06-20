import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.font_manager as fm


def generate_analysis_report(csv_filename="my_token_costs.csv"):
    """
    读取 Token 消耗 CSV 文件，生成综合可视化报告并保存为 PNG 图片。
    自动处理中文乱码问题。

    参数:
        csv_filename (str): CSV 文件路径，默认 'my_token_costs.csv'

    生成文件:
        - token_analysis_report.png  （四合一综合图表）
        - token_correlation_heatmap.png （相关性热力图）
    """
    # ---------- 第一步：解决中文乱码（跨平台字体设置） ----------
    # 查找系统可用的中文字体（按优先级）
    font_candidates = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    chosen_font = None
    for font in font_candidates:
        if font in available_fonts:
            chosen_font = font
            break

    if chosen_font is None:
        # 如果都没找到，尝试手动指定一个可能存在的路径（仅Linux）
        try:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        except:
            pass
        print("⚠️ 未找到常见中文字体，图表中文可能显示为方框。请安装中文字体。")
    else:
        plt.rcParams['font.sans-serif'] = [chosen_font]
        print(f"✅ 使用字体: {chosen_font}")

    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

    # ---------- 第二步：数据读取与检查 ----------
    if not os.path.exists(csv_filename):
        print(f"❌ 文件 {csv_filename} 不存在，请先写入数据。")
        return

    df = pd.read_csv(csv_filename, encoding='utf-8')
    if df.empty:
        print("❌ CSV 文件为空，无法生成图表。")
        return

    df['时间'] = pd.to_datetime(df['时间'])

    # 打印基础统计（控制台输出）
    print("\n" + "=" * 50)
    print("📊 数据概览（前5行）:")
    print(df.head())
    print("\n📈 描述性统计:")
    print(df[['原始输入消耗', '命中缓存量', '输出消耗']].describe())
    print("=" * 50 + "\n")

    # ---------- 第三步：绘制四合一报告 ----------
    sns.set_theme(style="whitegrid")
    # 同时设置 seaborn 字体（确保一致）
    sns.set(font=chosen_font if chosen_font else 'sans-serif')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Token 消耗综合报告', fontsize=20, fontweight='bold')

    # 子图1：时间趋势折线图
    ax1 = axes[0, 0]
    ax1.plot(df['时间'], df['原始输入消耗'], marker='o', label='输入消耗', linewidth=2)
    ax1.plot(df['时间'], df['命中缓存量'], marker='s', label='缓存命中', linewidth=2)
    ax1.plot(df['时间'], df['输出消耗'], marker='^', label='输出消耗', linewidth=2)
    ax1.set_title('① Token 消耗趋势（按时间）', fontsize=14)
    ax1.set_xlabel('时间')
    ax1.set_ylabel('Token 数量')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=30)

    # 子图2：各角色消耗构成（堆叠柱状图）
    ax2 = axes[0, 1]
    role_sum = df.groupby('角色')[['原始输入消耗', '命中缓存量', '输出消耗']].sum()
    role_sum.plot(kind='bar', stacked=True, ax=ax2,
                  color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_title('② 各角色总消耗构成', fontsize=14)
    ax2.set_xlabel('角色')
    ax2.set_ylabel('Token 总量')
    ax2.legend(title='消耗类型')
    ax2.tick_params(axis='x', rotation=0)

    # 子图3：Top 10 高消耗任务（水平条形图）
    ax3 = axes[1, 0]
    task_rank = df.groupby('任务')['原始输入消耗'].sum().sort_values(ascending=False).head(10)
    if not task_rank.empty:
        task_rank.plot(kind='barh', ax=ax3, color='coral')
        ax3.set_title('③ Top 10 高输入消耗任务', fontsize=14)
        ax3.set_xlabel('输入 Token 总消耗')
        ax3.invert_yaxis()
    else:
        ax3.text(0.5, 0.5, '无任务数据', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('③ Top 10 高输入消耗任务', fontsize=14)

    # 子图4：输出消耗分布（直方图 + KDE）
    ax4 = axes[1, 1]
    sns.histplot(df['输出消耗'], kde=True, ax=ax4, color='purple', alpha=0.6, bins=15)
    ax4.set_title('④ 输出消耗分布', fontsize=14)
    ax4.set_xlabel('输出 Token 数量')
    ax4.set_ylabel('频次')

    plt.tight_layout()
    report_img = "token_analysis_report.png"
    plt.savefig(report_img, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 综合报告已保存为: {report_img}")

    # # ---------- 额外：相关性热力图（单独保存） ----------
    # plt.figure(figsize=(8, 6))
    # corr = df[['原始输入消耗', '命中缓存量', '输出消耗']].corr()
    # sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True, linewidths=0.5)
    # plt.title('Token 消耗指标相关性热力图')
    # plt.tight_layout()
    # heatmap_img = "token_correlation_heatmap.png"
    # plt.savefig(heatmap_img, dpi=300)
    # plt.close()
    # print(f"✅ 相关性热力图已保存为: {heatmap_img}")

if __name__ == "__main__":
    generate_analysis_report()