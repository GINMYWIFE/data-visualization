# -*- coding: utf-8 -*-
"""价格文本三分类重标定 + 重跑预处理"""
import pandas as pd, numpy as np, re, json, io, os
from price_parser import parse_amount, parse_pct

df = pd.read_csv("steam_games.csv", encoding="utf-8", low_memory=False)
df = df.rename(columns={"Unnamed: 0": "row_id"})

def parse_num(s):
    # 改用 price_parser v2：修复年份误抽取(2,021€伪值)、零价式(24,--€)、复合串取错金额
    v = parse_amount(s, take="last")
    return np.nan if v is None else v

df["price_eur"] = df["price"].map(parse_num)

# ---------- 文本型价格的语义三分类 ----------
FREE_PAT = re.compile(
    r"free to play|free to download|gratuito|gratis|descargar|"
    r"\bfree\b|¡juega gratis|juega gratis|play for free|try .* for free|"
    r"free demo|free trial", re.I)
DEMO_PAT = re.compile(
    r"demo|trial|playable|version|prueba|试玩|beta version|"
    r"click here|click the button|play now|try it now|try the demo|"
    r"try before you buy|friend.?s? pass|handpass|buddy pass|"
    r"for .* pass users only|only for|note:|your saved progress|"
    r"signifier|download|install|start your|get tasty|learn to play|"
    r"solicitar acceso|accede|request access|join the common|"
    r"play with us|sizeable|cat quest|defend the bunker|am[r]?d evil|"
    r"roadwarden|undying|uneasy|mortuary assistant|shadows of the tomb|"
    r"the last roman|summer islands|silence|dwarfheim|dry drowning|"
    r"tiny death star|eden|to the rescue|unsung|realpolitiks|"
    r"punhos de repúdio|hollow knight|valheim|rising hell|", re.I)
PACK_PAT = re.compile(r"pack|paquete|pack de|datos del|información sobre", re.I)

def classify_price_text(s):
    """返回 'free' | 'demo_trial' | 'pack' | 'other_text' | None(数值或真缺失)"""
    if pd.isna(s): return None
    t = str(s).strip()
    if re.search(r"\d", t) and parse_num(t) is not None:
        return None
    if PACK_PAT.search(t):
        return "pack"
    if FREE_PAT.search(t):
        return "free"
    if DEMO_PAT.search(t):
        return "demo_trial"
    return "other_text"

df["price_status"] = df["price"].map(classify_price_text)
df["price_status"] = df["price_status"].where(
    df["price_status"].notna(),
    np.where(df["price_eur"].notna(), "numeric", "missing")
)

vc = df["price_status"].value_counts()
print(vc.to_string())

# 其它文本样例（需人工确认）
others = df.loc[df["price_status"]=="other_text","price"].value_counts()
print("\nother_text 样例:")
print(others.head(30).to_string())
print("other_text uniq:", len(others))

# DLC 交叉
print("\nprice_status x is_dlc:")
print(pd.crosstab(df["price_status"], df["is_dlc"]).to_string())

# 分类为 free/demo_trial 的样本，其 genre 分布
for st in ["free","demo_trial","pack"]:
    sub = df[df["price_status"]==st]
    print(f"\n{st} n={len(sub)} genre top: {sub['genre'].value_counts().head(4).to_dict()}")

# 写回供 preprocess 使用
df[["row_id","price_status"]].to_csv("price_status_map.csv", index=False, encoding="utf-8")

stats = {
    "price_status": vc.to_dict(),
    "other_text_samples": others.head(40).to_dict(),
    "other_text_uniq": int(len(others)),
    "crosstab": pd.crosstab(df["price_status"], df["is_dlc"]).to_dict(),
}
with io.open("price_status_report.json","w",encoding="utf-8") as f:
    json.dump(stats,f,ensure_ascii=False,indent=2,default=str)
print("\nDONE")
