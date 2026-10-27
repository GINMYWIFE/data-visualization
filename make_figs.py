# -*- coding: utf-8 -*-
"""初步可视化图表（数据准备阶段佐证图）"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["figure.facecolor"] = "white"

clean = pd.read_csv("steam_clean.csv", encoding="utf-8", low_memory=False)
raw = pd.read_csv("steam_games.csv", encoding="utf-8", low_memory=False)
os.makedirs("figs", exist_ok=True)

C1, C2, C3 = "#2E5EAA", "#E8833A", "#4CAF7D"

# ---------- 图1：价格分布 + 箱线图 ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
p = clean["price_eur"].dropna()
axes[0].hist(p, bins=60, color=C1, edgecolor="white", linewidth=.4)
axes[0].axvline(p.median(), color="red", ls="--", lw=1.4, label=f"中位数 {p.median():.2f}€")
axes[0].axvline(62.99, color=C3, ls=":", lw=1.6, label="IQR上界 62.99€")
axes[0].set_title("图1a  Steam 游戏价格分布（n=9548）")
axes[0].set_xlabel("价格 (€)"); axes[0].set_ylabel("游戏数量")
axes[0].legend(fontsize=8)
axes[0].set_xlim(0, 300) 

axes[1].boxplot(p, vert=False, widths=.6, showfliers=True,
                flierprops=dict(marker="o", ms=3, mfc="red", mec="red", alpha=.45))
axes[1].set_title("图1b  价格箱线图（离群点为高价游戏）")
axes[1].set_xlabel("价格 (€)")
axes[1].set_yticks([])
plt.tight_layout(); plt.savefig("figs/fig1_price_distribution.png", bbox_inches="tight"); plt.close()

# ---------- 图2：价格段 vs 好评率 / 热度 ----------
band = clean.groupby("price_band", observed=True).agg(
    n=("sentiment_all_score","size"),
    mean_sent=("sentiment_all_score","mean"),
    med_rev=("all_reviews_count_n","median")).reset_index()
band = band[band["n"] >= 30]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
b = axes[0].bar(band["price_band"], band["mean_sent"], color=C1, edgecolor="white")
axes[0].set_ylim(3.0, 4.1)
axes[0].set_title("图2a  不同价格段的平均好评评分（5分制）")
axes[0].set_xlabel("价格区间"); axes[0].set_ylabel("平均好评评分")
for r, v in zip(b, band["mean_sent"]):
    axes[0].text(r.get_x()+r.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=9)

b2 = axes[1].bar(band["price_band"], np.log10(band["med_rev"]), color=C3, edgecolor="white")
axes[1].set_title("图2b  不同价格段的中位评论数（热度，log10）")
axes[1].set_xlabel("价格区间"); axes[1].set_ylabel("log10(中位评论数)")
for r, v in zip(b2, np.log10(band["med_rev"])):
    axes[1].text(r.get_x()+r.get_width()/2, v+0.03, f"{10**v:,.0f}", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig("figs/fig2_price_band.png", bbox_inches="tight"); plt.close()

# ---------- 图3：热度 vs 好评率 散点 ----------
s = clean.dropna(subset=["price_eur","sentiment_all_score","all_reviews_count_n"])
s = s[s["all_reviews_count_n"] > 0]
fig, ax = plt.subplots(figsize=(8, 5.4))
sc = ax.scatter(s["price_eur"], s["sentiment_all_score"], c=np.log10(s["all_reviews_count_n"]),
                s=14, cmap="viridis", alpha=.55, linewidths=0)
ax.set_xlabel("价格 (€)"); ax.set_ylabel("全部好评评分（1=褒贬不一, 5=极其好评）")
ax.set_title("图3  价格 × 好评度 × 热度（三维：X=价格，Y=好评度，颜色=评论数热度）")
ax.axhline(3, color="gray", ls="--", lw=.9, alpha=.6)
cb = fig.colorbar(sc, ax=ax); cb.set_label("log10(全部评论数) —— 热度")
plt.tight_layout(); plt.savefig("figs/fig3_price_sentiment_heat.png", bbox_inches="tight"); plt.close()

# ---------- 图4：类型维度 气泡图 ----------
g = clean.groupby("genre").agg(n=("name","size"), price=("price_eur","median"),
                               sent=("sentiment_all_score","mean"),
                               heat=("all_reviews_count_n","median")).query("n>=80")
fig, ax = plt.subplots(figsize=(9, 5.6))
sc = ax.scatter(g["price"], g["sent"], s=np.log10(g["heat"])*70, c=g["n"],
                cmap="YlOrRd", alpha=.72, edgecolors="white", linewidths=.8)
for name, r in g.iterrows():
    ax.annotate(name, (r["price"], r["sent"]), fontsize=8.5,
                xytext=(4,4), textcoords="offset points")
ax.set_xlabel("该类型价格中位数 (€)"); ax.set_ylabel("平均好评评分")
ax.set_title("图4  游戏类型：价格 × 好评度 × 热度（气泡大小=中位评论数，颜色=游戏数量）")
cb = fig.colorbar(sc, ax=ax); cb.set_label("该类型游戏数量")
plt.tight_layout(); plt.savefig("figs/fig4_genre_bubble.png", bbox_inches="tight"); plt.close()

# ---------- 图5：缺失值 & 缺失机制 ----------
miss = raw.isna().mean().mul(100).sort_values()
miss = miss[miss > 0]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].barh(range(len(miss)), miss.values, color=C2)
axes[0].set_yticks(range(len(miss))); axes[0].set_yticklabels(miss.index, fontsize=8.5)
axes[0].set_xlabel("缺失率 (%)"); axes[0].set_title("图5a  各字段缺失率")
for i, v in enumerate(miss.values):
    axes[0].text(v+.6, i, f"{v:.1f}%", va="center", fontsize=7.5)

# 缺失机制：all_reviews 缺失 vs DLC / 年代
miss_by_dlc = raw.assign(m=raw["all_reviews"].isna()).groupby("is_dlc")["m"].mean().mul(100)
axes[1].bar([str(i) for i in miss_by_dlc.index], miss_by_dlc.values,
            color=["#7FA8D9", C2], edgecolor="white")
axes[1].set_xlabel("is_dlc（DLC/资料片 vs 主游戏）"); axes[1].set_ylabel("全部评论缺失率 (%)")
axes[1].set_title("图5b  缺失机制：资料片缺失率显著更高（系统性缺失）")
for i, v in enumerate(miss_by_dlc.values):
    axes[1].text(i, v+1.5, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout(); plt.savefig("figs/fig5_missing.png", bbox_inches="tight"); plt.close()

# ---------- 图6：Metacritic vs 好评度 + 年度趋势 ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
m = clean.dropna(subset=["metacritic_score","sentiment_all_score"])
axes[0].scatter(m["metacritic_score"], m["sentiment_all_score"], s=10, alpha=.4,
                color=C3, edgecolors="none")
z = np.polyfit(m["metacritic_score"], m["sentiment_all_score"], 1)
xs = np.linspace(m["metacritic_score"].min(), m["metacritic_score"].max(), 10)
axes[0].plot(xs, np.polyval(z, xs), color="red", lw=2,
             label=f"拟合线 r={m['metacritic_score'].corr(m['sentiment_all_score']):.3f}")
axes[0].set_xlabel("Metacritic 评分"); axes[0].set_ylabel("Steam 全部好评评分")
axes[0].set_title("图6a  媒体评分 vs 玩家好评（r=0.408）"); axes[0].legend(fontsize=9)

yr = clean.groupby("release_year").agg(n=("name","size"),
                                       sent=("sentiment_all_score","mean")).query("n>=20")
axes[1].plot(yr.index, yr["sent"], marker="o", ms=4, color=C1, label="平均好评评分")
ax2 = axes[1].twinx()
ax2.bar(yr.index, yr["n"], alpha=.18, color=C3, label="游戏数量")
axes[1].set_xlabel("发行年份"); axes[1].set_ylabel("平均好评评分", color=C1)
ax2.set_ylabel("当年游戏数量", color=C3)
axes[1].set_title("图6b  按发行年份：游戏数量激增但平均好评度下滑")
h1,l1 = axes[1].get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
axes[1].legend(h1+h2, l1+l2, fontsize=9, loc="lower left")
plt.tight_layout(); plt.savefig("figs/fig6_trend_score.png", bbox_inches="tight"); plt.close()

print("figs done:", os.listdir("figs"))
