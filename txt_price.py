# -*- coding: utf-8 -*-
import pandas as pd, re, io
from price_parser import parse_amount
r = pd.read_csv("steam_games.csv", encoding="utf-8", low_memory=False)

def pn(s):
    # 改用 price_parser v2：修复年份误抽取(2,021€伪值)、零价式(24,--€)、复合串取错金额
    v = parse_amount(s, take="last")
    return None if v is None else v

n = r["price"].map(pn)
mask = n.isna() & r["price"].notna()
txt = r.loc[mask, "price"].value_counts()
lines = []
for k, v in txt.items():
    sub = r[r["price"] == k]
    lines.append(f"{k!r}\tn={v}\tdlc={int(sub['is_dlc'].sum())}\tgenre={sub['genre'].value_counts().head(2).to_dict()}")
lines.append(f"TOTAL_TEXT\t{txt.sum()}")
lines.append(f"TOTAL_NA_PRICE\t{int(r['price'].isna().sum())}")
lines.append(f"NUMERIC\t{int((~mask & r['price'].notna()).sum())}")
io.open("txt_price.txt","w",encoding="utf-8").write("\n".join(lines))
print("ok", txt.sum())
