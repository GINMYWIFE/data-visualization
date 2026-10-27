# -*- coding: utf-8 -*-
"""溯源 discount_original_price 的异常值（Call of Duty 948.65€）"""
import pandas as pd, io

r = pd.read_csv("steam_games.csv", encoding="utf-8", low_memory=False)
L = []
L.append("="*95)
L.append("Call of Duty 系列原始三价格字段")
L.append("="*95)
cod = r[r["name"].str.contains("Call of Duty", na=False)]
for _, row in cod.head(20).iterrows():
    L.append(f"  {row['name'][:50]}")
    L.append(f"    price                  = {row['price']!r}")
    L.append(f"    discount_original_price= {row['discount_original_price']!r}")
    L.append(f"    discount_final_price   = {row['discount_final_price']!r}")
    L.append(f"    discount               = {row['discount']!r}")

L.append("")
L.append("="*95)
L.append("discount_original_price 字段的所有去重取值（前 40）")
L.append("="*95)
vc = r["discount_original_price"].value_counts()
for k, v in vc.head(40).items():
    L.append(f"  {v:>5}x  {k!r}")

L.append("")
L.append("="*95)
L.append("discount_final_price 字段的所有去重取值（前 40）")
L.append("="*95)
vc2 = r["discount_final_price"].value_counts()
for k, v in vc2.head(40).items():
    L.append(f"  {v:>5}x  {k!r}")

L.append("")
L.append("="*95)
L.append("discount 字段所有去重取值（前 40）")
L.append("="*95)
vc3 = r["discount"].value_counts()
for k, v in vc3.head(40).items():
    L.append(f"  {v:>5}x  {k!r}")

L.append("")
L.append("="*95)
L.append("是否存在 discount 字段为金额、且 == discount_original_price 的情况")
L.append("="*95)
mask = r["discount"].astype(str).str.contains("€", na=False)
L.append(f"discount 含 € 的行数: {int(mask.sum())}")
for _, row in r[mask].head(15).iterrows():
    L.append(f"  discount={row['discount']!r} orig={row['discount_original_price']!r} "
             f"final={row['discount_final_price']!r} price={row['price']!r} | {row['name'][:40]}")

io.open("orig_price_audit.txt","w",encoding="utf-8").write("\n".join(L))
print("ok")
