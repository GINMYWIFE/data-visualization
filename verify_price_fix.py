# -*- coding: utf-8 -*-
"""验证修复后的价格分布，并检查仍存在的可疑高价"""
import pandas as pd, numpy as np, io

c = pd.read_csv("steam_clean.csv", encoding="utf-8", low_memory=False)
r = pd.read_csv("steam_games.csv", encoding="utf-8", low_memory=False)

L = []
L.append("="*95)
L.append("A. 修复后 price_eur 分布")
L.append("="*95)
L.append(c["price_eur"].describe(percentiles=[.01,.05,.25,.5,.75,.9,.95,.99]).round(2).to_string())
L.append("")
L.append("零价格行数 (price_eur==0): %d" % int((c["price_eur"]==0).sum()))
L.append("其中 price_status 分布:")
L.append(c.loc[c["price_eur"]==0, "price_status"].value_counts().to_string())

L.append("")
L.append("="*95)
L.append("B. price_eur >= 100 的全部记录（修复后）")
L.append("="*95)
hi = c[c["price_eur"]>=100].sort_values("price_eur", ascending=False)
for _, row in hi.iterrows():
    L.append(f"  price={row['price_eur']:>9} list={row['price_list_eur']:>9} status={row['price_status']:<10} "
             f"| {str(row['name'])[:48]}")
L.append(f"  合计 {len(hi)} 条")

L.append("")
L.append("="*95)
L.append("C. price_list_eur >= 200 的（原价口径）")
L.append("="*95)
hl = c[c["price_list_eur"]>=200].sort_values("price_list_eur", ascending=False)
for _, row in hl.head(25).iterrows():
    L.append(f"  list={row['price_list_eur']:>9} price={row['price_eur']:>9} disc={row['discount_pct']} "
             f"| {str(row['name'])[:52]}")
L.append(f"  合计 {len(hl)} 条")

L.append("")
L.append("="*95)
L.append("D. price_list_eur 最大值溯源")
L.append("="*95)
top = c.nlargest(6, "price_list_eur")
for _, row in top.iterrows():
    L.append(f"\n  list={row['price_list_eur']}  name={row['name']}")
    L.append(f"  genre={row['genre']} is_dlc={row['is_dlc']} dev={row['developer']}")
    L.append(f"  price_status={row['price_status']} price_eur={row['price_eur']}")

L.append("")
L.append("="*95)
L.append("E. 价格档位表（原价口径）")
L.append("="*95)
t = c.groupby("price_band", observed=True).agg(
    n=("name","size"),
    mean_sent=("sentiment_all_score","mean"),
    med_rev=("all_reviews_count_n","median"),
    med_list=("price_list_eur","median")).reset_index()
t["price_band"]=t["price_band"].astype(str)
for _, row in t.iterrows():
    L.append(f"  {row['price_band']:<10} n={int(row['n']):>6}  mean_sent={row['mean_sent']:.3f}  "
             f"med_reviews={row['med_rev']:>10}  med_list_price={row['med_list']}")

L.append("")
L.append("="*95)
L.append("F. 订阅型商品样本")
L.append("="*95)
sub = c[c["is_subscription"]]
L.append(f"数量: {len(sub)}")
for _, row in sub.head(12).iterrows():
    L.append(f"  price={row['price_eur']} list={row['price_list_eur']} | {str(row['name'])[:45]}")

L.append("")
L.append("="*95)
L.append("G. 订阅型 vs 非订阅型的口碑/热度对比")
L.append("="*95)
for label, grp in [("订阅型", sub), ("非订阅型", c[~c["is_subscription"]])]:
    L.append(f"  {label}: n={len(grp)}  mean_sent={grp['sentiment_all_score'].mean():.3f}  "
             f"med_reviews={grp['all_reviews_count_n'].median():.0f}")

io.open("verify_price_fix.txt","w",encoding="utf-8").write("\n".join(L))
print("ok")
