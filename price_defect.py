# -*- coding: utf-8 -*-
"""排查 price 字段的解析缺陷：误抽取、0,--€ 格式、-100% 格式、与折后价不一致"""
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

df["p"] = df["price"].map(parse_num)
df["orig"] = df["discount_original_price"].map(parse_num)
df["final"] = df["discount_final_price"].map(parse_num)
df["disc_txt"] = df["discount"].astype(str)

L = []
L.append("="*100)
L.append("A. price 字段中含字母的字符串（非纯金额）")
L.append("="*100)
alpha = df[df["price"].notna() & df["price"].astype(str).str.contains(r"[A-Za-z]")]
L.append(f"数量: {len(alpha)}")
L.append(f"其中解析出了数值的（误抽取）: {int(alpha['p'].notna().sum())}")
for _, r in alpha[alpha["p"].notna()].sort_values("p", ascending=False).head(30).iterrows():
    L.append(f"  p={r['p']:>10} | price='{r['price']}' | name={r['name'][:50]}")

L.append("")
L.append("="*100)
L.append("B. 含 '0,--€' 或 '--€' 的西语零价格式")
L.append("="*100)
zero = df[df["price"].astype(str).str.contains(r"--\s*€|0,--", na=False)]
L.append(f"数量: {len(zero)}")
for _, r in zero.head(20).iterrows():
    L.append(f"  price='{r['price']}' | 解析p={r['p']} | discount='{r['discount']}' | final='{r['discount_final_price']}' | name={r['name'][:45]}")

L.append("")
L.append("="*100)
L.append("C. price 与 discount_final_price 不一致的行（应完全相等）")
L.append("="*100)
m = df.dropna(subset=["p","final"]).copy()
m["d"] = (m["p"] - m["final"]).abs()
bad = m[m["d"] > 0.01]
L.append(f"可比对行数: {len(m)}   不一致: {len(bad)} ({len(bad)/max(len(m),1)*100:.2f}%)")
L.append(f"  其中 price>final（即 price 是折后价、final 存的是别的价）: {int((bad['p']>bad['final']).sum())}")
L.append(f"  其中 price<final: {int((bad['p']<bad['final']).sum())}")
for _, r in bad.nlargest(20, "p").iterrows():
    L.append(f"  p={r['p']:>9} final={r['final']:>9} orig='{r['discount_original_price']}' disc='{r['discount']}' | {r['name'][:40]}")

L.append("")
L.append("="*100)
L.append("D. 价格在 100-200 且折扣 -90% 以上 的可疑行（原价极低但标价高）")
L.append("="*100)
susp = df[(df["p"]>=100)&(df["p"]<1000)]
for _, r in susp.iterrows():
    L.append(f"  p={r['p']:>9} | orig='{r['discount_original_price']}' | final='{r['discount_final_price']}' | disc='{r['discount']}' | {r['name'][:45]}")

L.append("")
L.append("="*100)
L.append("E. price 字符串中不含 € 符号但含数字的行")
L.append("="*100)
noeur = df[df["price"].notna() & ~df["price"].astype(str).str.contains("€", na=False)]
L.append(f"数量: {len(noeur)}")
for _, r in noeur.head(25).iterrows():
    L.append(f"  price='{r['price']}' | p={r['p']} | name={r['name'][:45]}")

L.append("")
L.append("="*100)
L.append("F. discount 字段的异常取值（被当成金额的）")
L.append("="*100)
dd = df[df["discount"].notna() & ~df["discount"].astype(str).str.contains("%", na=False)]
L.append(f"数量: {len(dd)}")
for _, r in dd.head(20).iterrows():
    L.append(f"  discount='{r['discount']}' | price='{r['price']}' | name={r['name'][:45]}")

io.open("price_defect_report.txt","w",encoding="utf-8").write("\n".join(L))
print("ok")
