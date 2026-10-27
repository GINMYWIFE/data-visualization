# -*- coding: utf-8 -*-
"""排查高价异常值（>=100€）的具体来源"""
import pandas as pd, numpy as np, re, io

df = pd.read_csv("steam_games.csv", encoding="utf-8", low_memory=False)

def parse_num(s):
    if pd.isna(s): return np.nan
    m = re.search(r"(\d[\d.,]*)", str(s))
    if not m: return np.nan
    raw = m.group(1)
    if "," in raw and "." in raw: raw = raw.replace(".", "").replace(",", ".")
    elif "," in raw: raw = raw.replace(",", ".")
    try: return float(raw)
    except ValueError: return np.nan

df["price_eur"] = df["price"].map(parse_num)
df["orig_eur"] = df["discount_original_price"].map(parse_num)
df["final_eur"] = df["discount_final_price"].map(parse_num)

hi = df[df["price_eur"] >= 100].sort_values("price_eur", ascending=False)
lines = []
lines.append(f"price>=100 的记录数: {len(hi)}")
lines.append(f"price>=200 的记录数: {int((df['price_eur']>=200).sum())}")
lines.append(f"price>=500 的记录数: {int((df['price_eur']>=500).sum())}")
lines.append(f"price>=1000 的记录数: {int((df['price_eur']>=1000).sum())}")
lines.append("")
lines.append("="*100)
lines.append("TOP 25 高价记录完整信息")
lines.append("="*100)
for _, r in hi.head(25).iterrows():
    lines.append(f"\n--- price={r['price_eur']}  (原始 price='{r['price']}') ---")
    lines.append(f"  name        : {r['name']}")
    lines.append(f"  genre       : {r['genre']}   is_dlc={r['is_dlc']}")
    lines.append(f"  developer   : {r['developer']}")
    lines.append(f"  publisher   : {r['publisher']}")
    lines.append(f"  release_date: {r['release_date']}")
    lines.append(f"  orig_price  : {r['discount_original_price']}")
    lines.append(f"  final_price : {r['discount_final_price']}")
    lines.append(f"  discount    : {r['discount']}")
    lines.append(f"  all_reviews : {r['all_reviews']} / {r['all_reviews_count']}")
    lines.append(f"  metacritic  : {r['metacritic_score']}")
    lines.append(f"  website     : {r['website']}")
    lines.append(f"  short_desc  : {str(r['short_description'])[:120]}")
    lines.append(f"  game_content: {str(r['game_content'])[:400]}")

lines.append("")
lines.append("="*100)
lines.append("200-2100 区间分布")
lines.append("="*100)
for lo, hi_ in [(100,200),(200,300),(300,500),(500,1000),(1000,3000)]:
    n = int(((df["price_eur"]>=lo)&(df["price_eur"]<hi_)).sum())
    lines.append(f"  [{lo}, {hi_}) : {n}")

lines.append("")
lines.append("="*100)
lines.append("低价高评/高热度的头部爆款（对照组，评价好但价格正常）")
lines.append("="*100)
def parse_count(s):
    if pd.isna(s): return np.nan
    m = re.search(r"\d[\d.,]*", str(s))
    if not m: return np.nan
    raw = re.sub(r"[^\d]", "", m.group(0))
    return float(raw) if raw else np.nan
df["rc_all"] = df["all_reviews_count"].map(parse_count)
top = df.nlargest(12, "rc_all")
for _, r in top.iterrows():
    lines.append(f"  {r['rc_all']:>10,.0f} 条评论 | {r['price_eur']:>8}€ | {r['all_reviews']:<22} | {r['name']}")

io.open("outlier_report.txt","w",encoding="utf-8").write("\n".join(lines))
print("ok")
