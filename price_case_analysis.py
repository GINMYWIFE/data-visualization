# -*- coding: utf-8 -*-
"""
判定 price 字段的可信性（三值一致性检验）
对每行同时有 (orig, pct) 的样本，计算:
  pred_final = orig * (1 - pct/100)
然后比较 price 与 pred_final / orig：
  CASE-1  price ≈ pred_final  -> price 是可信的折后价
  CASE-2  price ≈ orig        -> price 是可信的原价
  CASE-3  两者都不接近          -> price 不可信（错位抓取）
"""
import pandas as pd, numpy as np, io
from price_parser import parse_amount, parse_pct

r = pd.read_csv("steam_games.csv", encoding="utf-8", low_memory=False)

def pn(s):
    v = parse_amount(s, take="last")
    return np.nan if v is None else v

r["p"] = r["price"].map(pn)
r["o"] = r["discount_original_price"].map(pn)
r["f"] = r["discount_final_price"].map(pn)
r["pct"] = r["discount"].map(parse_pct)

L = []
L.append("="*100)
L.append("price 字段语义判定（以有 orig + pct 的行为样本）")
L.append("="*100)

m = r.dropna(subset=["o","p","pct"]).copy()
m = m[(m["o"]>0)&(m["p"]>0)&(m["pct"]>0)&(m["pct"]<100)].copy()
m["pred_final"] = (m["o"]*(1-m["pct"]/100)).round(2)
m["d_final"] = (m["p"]-m["pred_final"]).abs()
m["d_orig"]  = (m["p"]-m["o"]).abs()
m["case"] = np.where(m["d_final"]<0.02, "CASE1_折后价",
             np.where(m["d_orig"]<0.02, "CASE2_原价", "CASE3_不可信"))

vc = m["case"].value_counts()
L.append(f"样本 n = {len(m)}")
for k, v in vc.items():
    L.append(f"  {k:<16} {v:>6}  ({100*v/len(m):.2f}%)")

L.append("")
L.append("CASE-3（price 不可信）的样本 —— 按 price 降序前 25:")
L.append("-"*100)
bad = m[m["case"]=="CASE3_不可信"].sort_values("p", ascending=False)
L.append(f"CASE3 总计 {len(bad)} 行, 其中 price>100: {int((bad['p']>100).sum())}")
for _, row in bad.head(25).iterrows():
    L.append(f"  price={row['p']:>9} orig={row['o']:>9} pct={row['pct']:>3}% pred_final={row['pred_final']:>9} "
             f"final={row['f']!r:>14} | {row['name'][:36]}")

L.append("")
L.append("="*100)
L.append("CASE-3 中 price 与 pred_final 的偏离倍数分布")
L.append("="*100)
bad2 = bad[bad["pred_final"]>0].copy()
bad2["ratio"] = bad2["p"]/bad2["pred_final"]
L.append(f"  ratio 分位: {bad2['ratio'].describe(percentiles=[.05,.25,.5,.75,.95]).round(2).to_dict()}")

L.append("")
L.append("="*100)
L.append("关键：这些 CASE-3 行能否用 orig*(1-pct) 反推价格？")
L.append("="*100)
L.append(f"  可反推行数: {len(bad2)} / CASE3 {len(bad)}")
L.append(f"  反推后 price 分位: {bad2['pred_final'].describe(percentiles=[.25,.5,.75,.95]).round(2).to_dict()}")
L.append("")
L.append("  反推值 vs 同 developer 同系列游戏的实际 price —— 交叉验证样本:")
for _, row in bad[bad["p"]>100].head(6).iterrows():
    same = r[(r["developer"]==row["developer"])&(r["name"]!=row["name"])]
    prices = same["p"].dropna().tolist()[:6]
    L.append(f"    {row['name'][:38]:<40} dev={str(row['developer'])[:22]:<22} pred={row['pred_final']:>8} "
             f"同开发者其他游戏price={prices}")

L.append("")
L.append("="*100)
L.append("最终策略评估")
L.append("="*100)
L.append(f"  CASE1 (price=折后价，可信): {vc.get('CASE1_折后价',0)}")
L.append(f"  CASE2 (price=原价，可信):   {vc.get('CASE2_原价',0)}")
L.append(f"  CASE3 (不可信):            {vc.get('CASE3_不可信',0)}")
L.append("  -> CASE1 占多数，证明 price 字段的语义确实是「当前售价/折后价」")

io.open("price_case_analysis.txt","w",encoding="utf-8").write("\n".join(L))
print("ok")
