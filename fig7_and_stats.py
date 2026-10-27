# -*- coding: utf-8 -*-
"""补充图7：免费/试玩/付费对比 + 最终统计核对"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, json, io

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["figure.facecolor"] = "white"

c = pd.read_csv("steam_clean.csv", encoding="utf-8", low_memory=False)
C1, C2, C3 = "#2E5EAA", "#E8833A", "#4CAF7D"

# ---- 图7：价格档位 × 好评评分箱线图 + 各档统计 ----
order = ["免费","0-5€","5-10€","10-20€","20-30€","30-50€","50-100€",">100€","试玩版","捆绑包"]
c["pb"] = pd.Categorical(c["price_band"], categories=order, ordered=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
data = [c.loc[c["pb"]==b, "sentiment_all_score"].dropna().values for b in order]
data = [d if len(d) > 0 else [np.nan] for d in data]
bp = axes[0].boxplot(data, patch_artist=True, widths=.55, showfliers=False)
colors = [C3 if b=="免费" else (C2 if b in("试玩版","捆绑包") else C1) for b in order]
for patch, col in zip(bp["boxes"], colors):
    patch.set_facecolor(col); patch.set_alpha(.6); patch.set_edgecolor("white")
axes[0].set_xticks(range(1, len(order)+1))
axes[0].set_xticklabels(order, rotation=30, ha="right", fontsize=8.5)
axes[0].set_ylabel("全部好评评分（1–5）")
axes[0].set_title("图7a  各价格档位的玩家好评评分分布（箱线图）")
axes[0].axhline(3, color="gray", ls="--", lw=.9, alpha=.6)

# 右图：各档位的好评评分均值 + 热度（评论数中位）
grp = c.groupby("pb", observed=True).agg(
    n=("name","size"),
    mean_sent=("sentiment_all_score","mean"),
    med_rev=("all_reviews_count_n","median")).reset_index()
grp = grp[grp["n"] >= 20].sort_values("pb")
b = axes[1].bar(range(len(grp)), grp["mean_sent"], color=C1, edgecolor="white")
axes[1].set_xticks(range(len(grp)))
axes[1].set_xticklabels(grp["pb"].astype(str), rotation=30, ha="right", fontsize=8.5)
axes[1].set_ylim(2.5, 4.2)
axes[1].set_ylabel("平均好评评分")
axes[1].set_title("图7b  各价格档位平均好评评分（n≥20）")
for r, v, n in zip(b, grp["mean_sent"], grp["n"]):
    axes[1].text(r.get_x()+r.get_width()/2, v+0.03, f"{v:.2f}\n(n={int(n)})", ha="center", fontsize=7.5)
plt.tight_layout(); plt.savefig("figs/fig7_free_vs_paid.png", bbox_inches="tight"); plt.close()

# ---- 最终统计核对 ----
s = {}
s["clean_shape"] = list(c.shape)
s["price_status"] = c["price_status"].value_counts().to_dict()
s["is_dlc_n"] = int(c["is_dlc"].sum())
s["genre_n"] = int(c["genre"].nunique())
s["dev_n"] = int(c["developer"].nunique())
s["pub_n"] = int(c["publisher"].nunique())
s["price_valid"] = int(c["price_eur"].notna().sum())
s["date_parsed"] = int(c["release_dt"].notna().sum())
s["year_min"] = int(c["release_year"].min()); s["year_max"] = int(c["release_year"].max())
s["all_reviews_n"] = int(c["all_reviews_count_n"].notna().sum())
s["recent_reviews_n"] = int(c["recent_reviews_count_n"].notna().sum())
s["metacritic_n"] = int(c["metacritic_score"].notna().sum())
s["dup_name"] = int(c["name"].duplicated().sum())
s["outlier_n"] = int(c["is_price_outlier"].sum())
s["tags_n"] = int(c["tag_count"].sum())

for a,b2 in [(1984,1999),(2000,2009),(2010,2014),(2015,2019),(2020,2023)]:
    s[f"year_{a}_{b2}"] = int(c["release_year"].between(a,b2).sum())

# 分档表
t = c.groupby("pb", observed=True).agg(
    n=("name","size"), mean_sent=("sentiment_all_score","mean"),
    med_rev=("all_reviews_count_n","median")).reset_index()
t["pb"] = t["pb"].astype(str)
s["band_table"] = t.to_dict("records")

# 类型表
g = c.groupby("genre").agg(n=("name","size"), price=("price_eur","median"),
    sent=("sentiment_all_score","mean"), heat=("all_reviews_count_n","median")).query("n>=80").reset_index()
s["genre_table"] = g.sort_values("n", ascending=False).to_dict("records")

# DLC vs 主游戏 缺失率对比（原始表）
r = pd.read_csv("steam_games.csv", encoding="utf-8", low_memory=False)
for col in ["all_reviews","short_description","metacritic_score","price"]:
    t2 = r.groupby("is_dlc")[col].apply(lambda x: x.isna().mean()*100)
    s[f"miss_{col}"] = {"dlc": round(float(t2[True]),2), "main": round(float(t2[False]),2)}

# 相关性
sub = c.dropna(subset=["price_eur","sentiment_all_score","all_reviews_count_n"])
sub = sub[sub["all_reviews_count_n"]>0]
s["corr"] = {
 "price_vs_sent": round(float(sub["price_eur"].corr(sub["sentiment_all_score"])),4),
 "price_vs_log_rev": round(float(np.log1p(sub["price_eur"]).corr(np.log1p(sub["all_reviews_count_n"]))),4),
 "logrev_vs_sent": round(float(np.log1p(sub["all_reviews_count_n"]).corr(sub["sentiment_all_score"])),4),
}
m2 = c.dropna(subset=["metacritic_score","sentiment_all_score"])
s["corr"]["metacritic_vs_sent"] = round(float(m2["metacritic_score"].corr(m2["sentiment_all_score"])),4)
s["corr"]["price_vs_metacritic"] = round(float(c.dropna(subset=["price_eur","metacritic_score"])["price_eur"].corr(c.dropna(subset=["price_eur","metacritic_score"])["metacritic_score"])),4)

# 分数分布
s["sent_dist"] = c["sentiment_all_score"].value_counts().sort_index().to_dict()
s["rc_all_stats"] = c["all_reviews_count_n"].describe(percentiles=[.25,.5,.75,.9,.99]).round(1).to_dict()

with io.open("final_stats.json","w",encoding="utf-8") as f:
    json.dump(s,f,ensure_ascii=False,indent=2,default=str)
print("ok")
