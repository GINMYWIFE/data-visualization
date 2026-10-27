# -*- coding: utf-8 -*-
"""排查 discount_original_price 中的伪值（原价为抓取页面错位产生的值）"""
import pandas as pd, numpy as np, io
from price_parser import parse_amount

r = pd.read_csv("steam_games.csv", encoding="utf-8", low_memory=False)

def pn(s):
    v = parse_amount(s, take="last")
    return np.nan if v is None else v

r["p"]  = r["price"].map(pn)
r["o"]  = r["discount_original_price"].map(pn)
r["f"]  = r["discount_final_price"].map(pn)

L = []
L.append("="*95)
L.append("A. discount_original_price 的量级分布")
L.append("="*95)
L.append(r["o"].describe(percentiles=[.5,.9,.95,.99,.999]).round(2).to_string())
L.append("")
L.append("超过 200€ 的原价行数: %d" % int((r["o"]>200).sum()))
L.append("超过 500€ 的原价行数: %d" % int((r["o"]>500).sum()))
L.append("超过 1000€ 的原价行数: %d" % int((r["o"]>1000).sum()))

L.append("")
L.append("="*95)
L.append("B. 超高原价行：是否同时缺失 discount_final_price（=无法交叉验证）")
L.append("="*95)
hi = r[r["o"]>200]
L.append(f"总计 {len(hi)} 行")
L.append(f"  其中 discount_final_price 缺失: {int(hi['f'].isna().sum())}")
L.append(f"  其中 discount_final_price 存在: {int(hi['f'].notna().sum())}")
L.append("")
L.append("按 (orig值) 分组，看是否同一数值被大量不同游戏共享（页面错位的典型特征）:")
vc = hi["discount_original_price"].value_counts()
for k, v in vc.head(15).items():
    names = hi[hi["discount_original_price"]==k]["name"].head(4).tolist()
    L.append(f"  {k!r:18} x{v:<4} 例: {names}")

L.append("")
L.append("="*95)
L.append("C. 自洽性检验：orig / price 比值分布（折后价应 >= 原价*0.5 才合理）")
L.append("="*95)
m = r.dropna(subset=["o","p"]).copy()
m = m[(m["o"]>0) & (m["p"]>0)]
m["ratio"] = m["o"] / m["p"]
L.append("比值分位: " + str(m["ratio"].quantile([.5,.9,.95,.99,.999]).round(2).to_dict()))
L.append(f"比值 > 2 的行数: {int((m['ratio']>2).sum())}  ({len(m)})")
L.append(f"比值 > 5 的行数: {int((m['ratio']>5).sum())}")
L.append("")
L.append("比值 > 5 的样本:")
for _, row in m.nlargest(20, "ratio").iterrows():
    L.append(f"  orig={row['o']:>9} price={row['p']:>8} ratio={row['ratio']:>7.1f} "
             f"disc={row['discount']!r} final={row['discount_final_price']!r} | {row['name'][:38]}")

L.append("")
L.append("="*95)
L.append("D. 有 orig 无 final（无法交叉验证）vs 有 orig 有 final（可验证）的量级对比")
L.append("="*95)
nof = m[m["f"].isna()]; wf = m[m["f"].notna()]
L.append(f"  orig有、final无: n={len(nof)}  orig中位数={nof['o'].median():.2f}  "
         f"orig>200占比={100*(nof['o']>200).mean():.2f}%")
L.append(f"  orig有、final有: n={len(wf)}  orig中位数={wf['o'].median():.2f}  "
         f"orig>200占比={100*(wf['o']>200).mean():.2f}%")

L.append("")
L.append("="*95)
L.append("E. 用可验证子集推断规则：orig 缺失时能否用 price 反推原价")
L.append("="*95)
wf2 = wf.copy()
from price_parser import parse_pct
wf2["pct"] = wf2["discount"].map(parse_pct)
wf2["expected_final"] = (wf2["o"]*(1-wf2["pct"]/100)).round(2)
ok = wf2.dropna(subset=["expected_final"])
L.append(f"  用 orig*(1-pct) 预测 final 的样本 n={len(ok)}, "
         f"吻合率={100*(ok['expected_final']-ok['f']).abs().lt(0.02).mean():.2f}%")

io.open("orig_price_flaw.txt","w",encoding="utf-8").write("\n".join(L))
print("ok")
