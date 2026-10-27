# -*- coding: utf-8 -*-
"""
Steam 游戏数据清洗与预处理脚本（数据准备阶段最终版）
输入: steam_games.csv
输出: steam_clean.csv, steam_tags_long.csv, prep_summary.json
"""
import pandas as pd, numpy as np, re, json, io, os
from price_parser import parse_amount, parse_pct, parse_count as _pc, is_subscription

# ============ 1. 读取 ============
df = pd.read_csv("steam_games.csv", encoding="utf-8", low_memory=False)
df = df.rename(columns={"Unnamed: 0": "row_id"})
RAW_ROWS, RAW_COLS = df.shape
print("原始:", df.shape)

# ============ 2. 字段解析工具 ============
ES_MONTH = {
    "ENE":"ENE","ABR":"ABR","MAR":"MAR","ABR":"ABR","MAY":"MAY","JUN":"JUN",
    "JUL":"JUL","AGO":"AGO","SEP":"SEP","OCT":"OCT","NOV":"NOV","DIC":"DIC",
    "ENE":"ENE","FEB":"FEB","ABR":"ABR","MAY":"MAY","JUN":"JUN","JUL":"JUL",
    "AGO":"AGO","SEP":"SEP","OCT":"OCT","NOV":"NOV","DIC":"DIC",
}
MONTH_MAP = {
    "ENE":1,"FEB":2,"MAR":3,"ABR":4,"MAY":5,"JUN":6,
    "JUL":7,"AGO":8,"SEP":9,"OCT":10,"NOV":11,"DIC":12,
    "JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
    "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12,
}

def parse_date_es(s):
    """解析 '9 MAR 2021' / '12 ENE 2022' / 'Mar 9, 2021'"""
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    if not s:
        return pd.NaT
    # 形式1: D MON YYYY
    m = re.match(r"^(\d{1,2})\s+([A-Za-zÁÉÍÓÚáéíóú]{3,})\s+(\d{4})$", s)
    if m:
        d, mon, y = int(m.group(1)), m.group(2).upper()[:3], int(m.group(3))
        if mon in MONTH_MAP:
            return pd.to_datetime(f"{y}-{MONTH_MAP[mon]:02d}-{d:02d}", errors="coerce")
    # 形式2: MON D, YYYY
    m = re.match(r"^([A-Za-zÁÉÍÓÚáéíóú]{3,})\s+(\d{1,2}),?\s+(\d{4})$", s)
    if m:
        mon, d, y = m.group(1).upper()[:3], int(m.group(2)), int(m.group(3))
        if mon in MONTH_MAP:
            return pd.to_datetime(f"{y}-{MONTH_MAP[mon]:02d}-{d:02d}", errors="coerce")
    # 形式3: YYYY-MM-DD
    try:
        return pd.to_datetime(s, format="%Y-%m-%d")
    except Exception:
        return pd.NaT

def parse_num(s):
    """西语金额 -> float（走 price_parser v2，已修复年份误抽取/零价式/复合串）"""
    v = parse_amount(s, take="last")
    return np.nan if v is None else v

def parse_num_first(s):
    """取串中第一个金额（用于订阅型商品的最低门槛价）"""
    v = parse_amount(s, take="first")
    return np.nan if v is None else v

def parse_count(s):
    """'(3,113)' -> 3113"""
    v = _pc(s)
    return np.nan if v is None else float(v)

SENT_SCORE = {
    "Extremadamente positivas": 5, "Muy positivas": 4, "Mayormente positivas": 3,
    "Positivas": 2, "Variadas": 1.5, "Mixtas": 1.5,
    "Mayormente negativas": 1, "Mayoritariamente negativas": 1,
    "Muy negativas": 0.5, "Extremadamente negativas": 0, "Negativas": 1,
}
SENT_PCT = {  # 好评率近似区间中点（用于转成百分比）
    "Extremadamente positivas": 0.95, "Muy positivas": 0.90, "Mayormente positivas": 0.80,
    "Positivas": 0.70, "Variadas": 0.50, "Mixtas": 0.50,
    "Mayormente negativas": 0.25, "Mayoritariamente negativas": 0.25,
    "Muy negativas": 0.10, "Extremadamente negativas": 0.05, "Negativas": 0.20,
}

def map_sent(s, table):
    if pd.isna(s):
        return np.nan
    return table.get(str(s).strip(), np.nan)

# ============ 3. 衍生字段抽取 ============
df["release_dt"]   = df["release_date"].map(parse_date_es)
df["release_year"] = df["release_dt"].dt.year
df["release_month"]= df["release_dt"].dt.month
df["release_year_month"] = df["release_dt"].dt.to_period("M").astype(str)

df["price_eur"]       = df["price"].map(parse_num)
df["price_orig_eur"]  = df["discount_original_price"].map(parse_num)
df["price_final_eur"] = df["discount_final_price"].map(parse_num)
# discount 字段混合了百分比与金额：用 parse_pct 只取百分比，金额另行处理
df["discount_pct"]    = df["discount"].map(parse_pct)
# 订阅型商品（按月/按年续订）的 price 是促销文案，取最低门槛价
df["is_subscription"] = df["price"].map(is_subscription)
df.loc[df["is_subscription"], "price_eur"] = df.loc[df["is_subscription"], "price"].map(parse_num_first)
# price 字段语义 = 当前售价（打折时即折后价）。
# 原价（list price）只有能被 discount_final_price 交叉验证时才可信：
#   有 final 时 orig*(1-pct) 与 final 吻合率 99.14%；
#   无 final 时该吻合率仅 0.24%，且伪 orig 被多款无关游戏共享（如 948,65€ x15 全是 CoD），
#   属抓取时页面结构错位产生的伪值 -> 不可信，回退为当前售价。
df["orig_verified"] = (
    df["price_orig_eur"].notna()
    & df["price_final_eur"].notna()
    & df["discount_pct"].notna()
)
df["price_list_eur"] = df["price_orig_eur"].where(df["orig_verified"], df["price_eur"])

df["recent_reviews_count_n"] = df["recent_reviews_count"].map(parse_count)
df["all_reviews_count_n"]    = df["all_reviews_count"].map(parse_count)

df["sentiment_recent_score"] = df["recent_reviews"].map(lambda s: map_sent(s, SENT_SCORE))
df["sentiment_all_score"]    = df["all_reviews"].map(lambda s: map_sent(s, SENT_SCORE))
df["sentiment_recent_pct"]   = df["recent_reviews"].map(lambda s: map_sent(s, SENT_PCT))
df["sentiment_all_pct"]      = df["all_reviews"].map(lambda s: map_sent(s, SENT_PCT))

# 价格文本三分类（复用 price_reclass.py 的规则）
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
PACK_PAT = re.compile(r"pack|paquete|datos del|información sobre", re.I)

def classify_price_text(s):
    if pd.isna(s):
        return "missing"
    t = str(s).strip()
    if re.search(r"\d", t) and not pd.isna(parse_num(t)):
        return "numeric"
    if PACK_PAT.search(t):
        return "pack"
    if FREE_PAT.search(t):
        return "free"
    if DEMO_PAT.search(t):
        return "demo_trial"
    return "other_text"

df["price_status"] = df["price"].map(classify_price_text)

# 是否免费（仅 free 类；demo_trial 单独标注）
df["is_free"] = df["price_status"].eq("free")
df["is_demo"] = df["price_status"].eq("demo_trial")

# 热��指标
df["popularity_log"] = np.log1p(df["all_reviews_count_n"])
df["recent_popularity_log"] = np.log1p(df["recent_reviews_count_n"])

# 价格分层（以原价 list price 为口径，避免打折商品被系统性低估）
df["price_band"] = pd.cut(df["price_list_eur"], [0,5,10,20,30,50,100,np.inf],
                          labels=["0-5€","5-10€","10-20€","20-30€","30-50€","50-100€",">100€"])
# 免费/试玩/捆绑包独立成档，不与 0-5€ 混淆
df["price_band"] = df["price_band"].astype(object)
df.loc[df["is_free"], "price_band"] = "免费"
df.loc[df["is_demo"], "price_band"] = "试玩版"
df.loc[df["price_status"].eq("pack"), "price_band"] = "捆绑包"
df["price_band"] = pd.Categorical(df["price_band"],
    categories=["免费","0-5€","5-10€","10-20€","20-30€","30-50€","50-100€",">100€","试玩版","捆绑包"],
    ordered=True)

# 发行年代
def era(y):
    if pd.isna(y): return "未知"
    if y < 2000: return "1990前"
    if y < 2010: return "2000-2009"
    if y < 2015: return "2010-2014"
    if y < 2020: return "2015-2019"
    return "2020后"
df["release_era"] = df["release_year"].map(era)

# 标签数量
df["tag_count"] = df["tags"].fillna("").apply(lambda s: len([t for t in str(s).split(",") if t.strip()]))

# ============ 4. 数据质量报告 ============
summary = {}
summary["raw_rows"] = int(RAW_ROWS)
summary["raw_cols"] = int(RAW_COLS)
summary["date_parsed"] = int(df["release_dt"].notna().sum())
summary["date_unparsed"] = int(df["release_dt"].isna().sum())
summary["date_range"] = [str(df["release_dt"].min().date()), str(df["release_dt"].max().date())]
summary["year_dist"] = df["release_year"].value_counts().sort_index().to_dict()
summary["price_numeric_n"] = int(df["price_eur"].notna().sum())
summary["price_status"] = df["price_status"].value_counts().to_dict()
summary["is_free_n"] = int(df["is_free"].sum())
summary["is_demo_n"] = int(df["is_demo"].sum())
summary["price_true_na"] = int(df["price"].isna().sum())
summary["dup_by_name"] = int(df.duplicated(subset=["name"]).sum())
summary["dup_full"] = int(df.duplicated().sum())
summary["sent_all_unparsed"] = int((df["all_reviews"].notna() & df["sentiment_all_score"].isna()).sum())
summary["sent_recent_unparsed"] = int((df["recent_reviews"].notna() & df["sentiment_recent_score"].isna()).sum())
summary["recent_low_sample"] = int(df["recent_reviews"].astype(str).str.contains("reseña", na=False).sum())
summary["is_subscription_n"] = int(df["is_subscription"].sum())
summary["price_eur_max"] = float(df["price_eur"].max())
summary["price_eur_gt100_n"] = int((df["price_eur"] > 100).sum())
summary["price_list_max"] = float(df["price_list_eur"].max())
summary["price_list_gt100_n"] = int((df["price_list_eur"] > 100).sum())
summary["price_list_gt200_n"] = int((df["price_list_eur"] > 200).sum())
summary["orig_verified_n"] = int(df["orig_verified"].sum())
summary["orig_unverified_n"] = int(df["price_orig_eur"].notna().sum() - df["orig_verified"].sum())

# ============ 5. 清洗操作 ============
# 5.1 去重（按 name 保留首个）
before = len(df)
df = df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
summary["rows_after_dedup"] = int(len(df))
summary["rows_removed_dedup"] = int(before - len(df))

# 5.2 异常值：价格按 IQR 上界标记（不删除，标记 is_price_outlier）
# 基于 price_eur（当前售价，全量可信）；price_list_eur 仅在原价可验证时才有不同
p = df["price_eur"].dropna()
q1, q3 = p.quantile(.25), p.quantile(.75)
iqr = q3 - q1
upper = q3 + 1.5 * iqr
df["is_price_outlier"] = df["price_eur"] > upper
summary["price_iqr"] = {"q1": float(q1), "q3": float(q3), "upper_fence": float(upper),
                        "n_outlier": int(df["is_price_outlier"].sum())}
mu, sd = p.mean(), p.std()
summary["price_3sigma"] = {"mean": float(mu), "std": float(sd),
                           "upper": float(mu+3*sd), "n_outlier": int((p > mu+3*sd).sum())}

# 5.3 一致性校验
d = df.dropna(subset=["price_orig_eur","price_final_eur","discount_pct"])
d = d[(d["price_orig_eur"]>0) & (d["discount_pct"]>0) & (d["discount_pct"]<100)]
calc = (d["price_orig_eur"]*(1-d["discount_pct"]/100)).round(2)
summary["discount_consistency_rate"] = float((calc-d["price_final_eur"]).abs().lt(0.02).mean())
summary["n_discount_checked"] = int(len(d))
rv = df.dropna(subset=["recent_reviews_count_n","all_reviews_count_n"])
summary["review_logic_violations"] = int((rv["recent_reviews_count_n"]>rv["all_reviews_count_n"]).sum())
summary["reviews_anomally_true"] = int(df["reviews_anomally"].sum())

# 5.4 价格缺失填充口径
summary["price_fill_median"] = float(df["price_eur"].median())
df["price_eur_filled"] = df["price_eur"].fillna(df["price_eur"].median())
df["price_missing_flag"] = df["price_eur"].isna()

# ============ 6. 输出 ============
keep = ["row_id","name","is_dlc","is_free","is_demo","is_subscription","price_status","genre","developer","publisher","release_dt",
        "release_year","release_month","release_year_month","release_era",
        "price_eur","price_list_eur","price_orig_eur","price_final_eur","discount_pct","orig_verified","price_band",
        "price_eur_filled","price_missing_flag","is_price_outlier",
        "sentiment_recent_score","sentiment_all_score","sentiment_recent_pct","sentiment_all_pct",
        "recent_reviews_count_n","all_reviews_count_n",
        "popularity_log","recent_popularity_log","metacritic_score",
        "reviews_anomally","tag_count","tags","short_description"]
clean = df[keep].copy()
clean.to_csv("steam_clean.csv", index=False, encoding="utf-8")

# 标签长表（多值展开）
rows = []
for _, r in df.iterrows():
    if pd.isna(r["tags"]): continue
    for t in str(r["tags"]).split(","):
        t = t.strip()
        if t:
            rows.append({"name": r["name"], "genre": r["genre"], "tag": t,
                         "price_eur": r["price_eur"], "sentiment_all_score": r["sentiment_all_score"],
                         "all_reviews_count_n": r["all_reviews_count_n"]})
tags_long = pd.DataFrame(rows)
tags_long.to_csv("steam_tags_long.csv", index=False, encoding="utf-8")

summary["clean_shape"] = list(clean.shape)
summary["tags_long_shape"] = list(tags_long.shape)
summary["file_sizes_mb"] = {
    "steam_games.csv": round(pd.io.common.file_exists("steam_games.csv") and 0 or 0, 2)
}
import os
summary["file_sizes_mb"] = {f: round(os.path.getsize(f)/1024**2, 2)
    for f in ["steam_games.csv","steam_clean.csv","steam_tags_long.csv"]}

# 关键关联
sub = clean.dropna(subset=["price_eur","sentiment_all_score","all_reviews_count_n"])
sub = sub[sub["all_reviews_count_n"]>0]
summary["corr"] = {
    "price_vs_sentiment_all": round(float(sub["price_eur"].corr(sub["sentiment_all_score"])),4),
    "price_vs_log_reviews": round(float(np.log1p(sub["price_eur"]).corr(np.log1p(sub["all_reviews_count_n"]))),4),
    "log_reviews_vs_sentiment_all": round(float(np.log1p(sub["all_reviews_count_n"]).corr(sub["sentiment_all_score"])),4),
}
# 原价口径（list price）的相关性对比
subL = clean.dropna(subset=["price_list_eur","sentiment_all_score","all_reviews_count_n"])
subL = subL[subL["all_reviews_count_n"] > 0]
summary["corr"]["listprice_vs_sentiment_all"] = round(float(subL["price_list_eur"].corr(subL["sentiment_all_score"])),4)
summary["corr"]["listprice_vs_log_reviews"] = round(float(np.log1p(subL["price_list_eur"]).corr(np.log1p(subL["all_reviews_count_n"]))),4)
m2 = clean.dropna(subset=["metacritic_score","sentiment_all_score"])
summary["corr"]["metacritic_vs_sentiment_all"] = round(float(m2["metacritic_score"].corr(m2["sentiment_all_score"])),4)

band = sub.groupby("price_band", observed=True).agg(
    n=("sentiment_all_score","size"),
    mean_sent=("sentiment_all_score","mean"),
    median_reviews=("all_reviews_count_n","median")).reset_index()
band["price_band"] = band["price_band"].astype(str)
summary["price_band_table"] = band.to_dict("records")

with io.open("prep_summary.json","w",encoding="utf-8") as f:
    json.dump(summary,f,ensure_ascii=False,indent=2,default=str)
print("DONE")
