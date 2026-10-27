# -*- coding: utf-8 -*-
"""补充诊断：解析失败样例、评论类别映射、缺失机制分析"""
import pandas as pd, numpy as np, re, json, io
from price_parser import parse_amount

df = pd.read_csv("steam_games.csv", encoding="utf-8", low_memory=False)
res = {}

# --- 1. price 解析失败样例 ---
# 改用 price_parser v2：修复年份误抽取(2,021€伪值)、零价式(24,--€)、复合串取错金额
def parse_price(s):
    v = parse_amount(s, take="last")
    return np.nan if v is None else v

df["price_num"] = df["price"].map(parse_price)
fail = df[df["price"].notna() & df["price_num"].isna()]
res["price_parse_fail_n"] = int(len(fail))
res["price_parse_fail_samples"] = fail["price"].value_counts().head(10).to_dict()
res["price_fail_by_isdlc"] = fail["is_dlc"].value_counts().to_dict()
res["price_fail_by_genre"] = fail["genre"].value_counts().head(5).to_dict()

# price 缺失与 is_dlc / genre 关系
res["price_missing_by_isdlc"] = df[df["price"].isna()]["is_dlc"].value_counts().to_dict()
res["price_missing_by_genre_top"] = df[df["price"].isna()]["genre"].value_counts().head(6).to_dict()
res["price_nonnull_by_isdlc"] = df[df["price"].notna()]["is_dlc"].value_counts().to_dict()

# --- 2. release_date 未解析样例 ---
def parse_date(s):
    if pd.isna(s): return pd.NaT
    s = str(s).strip()
    for fmt in ("%d %b %Y", "%d %B %Y", "%b %d, %Y", "%Y-%m-%d", "%d %b %y"):
        try: return pd.to_datetime(s, format=fmt)
        except Exception: continue
    return pd.to_datetime(s, errors="coerce")

df["release_dt"] = df["release_date"].map(parse_date)
unp = df[df["release_date"].notna() & df["release_dt"].isna()]
res["date_unparsed_n"] = int(len(unp))
res["date_unparsed_samples"] = unp["release_date"].value_counts().head(15).to_dict()
res["date_unparsed_by_isdlc"] = unp["is_dlc"].value_counts().to_dict()
# 空字符串
res["date_blank_str"] = int((df["release_date"].fillna("").str.strip() == "").sum())
res["date_na_by_isdlc"] = df[df["release_date"].isna()]["is_dlc"].value_counts().to_dict()

# --- 3. 评论类别 -> 有序评分 ---
SENT = {
 "Extremadamente positivas": 5, "Muy positivas": 4, "Mayormente positivas": 3,
 "Positivas": 2, "Variadas": 1.5, "Mixtas": 1.5,
 "Mayoritariamente negativas": 0, "Mayormente negativas": 1,
 "Muy negativas": 0.5, "Extremadamente negativas": 0,
}
def sent(s):
    if pd.isna(s): return np.nan
    s = str(s).strip()
    if s in SENT: return SENT[s]
    return np.nan

df["sent_recent"] = df["recent_reviews"].map(sent)
df["sent_all"] = df["all_reviews"].map(sent)
res["sent_recent_unparsed_n"] = int(df["sent_recent"].isna().sum() - df["recent_reviews"].isna().sum())
res["sent_all_unparsed_n"] = int(df["sent_all"].isna().sum() - df["all_reviews"].isna().sum())
res["sent_recent_unparsed_samples"] = df[df["recent_reviews"].notna() & df["sent_recent"].isna()]["recent_reviews"].value_counts().head(10).to_dict()
res["sent_all_unparsed_samples"] = df[df["all_reviews"].notna() & df["sent_all"].isna()]["all_reviews"].value_counts().head(10).to_dict()

def parse_reviews(s):
    if pd.isna(s): return np.nan
    m = re.search(r"\d[\d.,]*", str(s))
    if not m: return np.nan
    raw = re.sub(r"[^\d]", "", m.group(0))
    return float(raw) if raw else np.nan

df["rc_recent"] = df["recent_reviews_count"].map(parse_reviews)
df["rc_all"] = df["all_reviews_count"].map(parse_reviews)
df["price_num"] = df["price_num"]
df["final_num"] = df["discount_final_price"].map(parse_price)
df["orig_num"] = df["discount_original_price"].map(parse_price)

# --- 4. 缺失机制：all_reviews 缺失 vs 价格/年份 ---
m = df[df["all_reviews"].isna()]
res["all_reviews_missing_by_isdlc"] = m["is_dlc"].value_counts().to_dict()
res["all_reviews_missing_year_dist"] = df["release_dt"].dt.year.value_counts(dropna=False).sort_index().tail(12).to_dict()
res["all_reviews_missing_price_missing_pct"] = round(float(m["price"].isna().mean()*100),2)
res["all_reviews_present_price_missing_pct"] = round(float(df[df["all_reviews"].notna()]["price"].isna().mean()*100),2)

# --- 5. 关联性（用有序评分） ---
sub = df.dropna(subset=["price_num","sent_all","rc_all"]).copy()
sub = sub[sub["rc_all"]>0]
res["n_corr_sample"] = int(len(sub))
res["corr"] = {
 "price_vs_sent_all": round(float(sub["price_num"].corr(sub["sent_all"])),4),
 "price_vs_sent_recent": round(float(sub.dropna(subset=['sent_recent']).pipe(lambda x: x['price_num'].corr(x['sent_recent']))),4),
 "log_rc_all_vs_sent_all": round(float(np.log1p(sub["rc_all"]).corr(sub["sent_all"])),4),
 "price_vs_log_rc_all": round(float(np.log1p(sub["price_num"]).corr(np.log1p(sub["rc_all"]))),4),
}
sub2 = df.dropna(subset=["metacritic_score","sent_all"])
res["corr"]["metacritic_vs_sent_all"] = round(float(sub2["metacritic_score"].corr(sub2["sent_all"])),4)
res["corr"]["price_vs_metacritic"] = round(float(df.dropna(subset=["price_num","metacritic_score"])["price_num"].corr(df.dropna(subset=["price_num","metacritic_score"])["metacritic_score"])),4)

# 分价格段好评率
bins = [0,5,10,20,30,50,100,1000]
sub["pband"] = pd.cut(sub["price_num"], bins)
_band = sub.groupby("pband", observed=True).agg(
    n=("sent_all","size"), mean_sent=("sent_all","mean"),
    mean_log_rc=("rc_all", lambda s: float(np.log1p(s).mean()))).round(3).reset_index()
_band["price_band"] = _band["pband"].astype(str)
res["price_band_sent_all"] = [
    {"price_band": r["price_band"], "n": int(r["n"]),
     "mean_sent": float(r["mean_sent"]), "mean_log_rc": float(r["mean_log_rc"])}
    for _, r in _band.iterrows()
]

# 热度（评论数）分布
res["rc_all_stats"] = df["rc_all"].describe(percentiles=[.25,.5,.75,.9,.99]).round(1).to_dict()
res["rc_recent_stats"] = df["rc_recent"].describe(percentiles=[.25,.5,.75,.9,.99]).round(1).to_dict()
res["rc_log10_decile"] = np.log10(df["rc_all"].dropna()).describe(percentiles=[.1,.25,.5,.75,.9]).round(3).to_dict()

# genre 维度聚合样例（用于方案）
res["genre_agg"] = df.groupby("genre").agg(
    n=("name","size"),
    price_med=("price_num","median"),
    rc_all_med=("rc_all","median"),
    sent_all_mean=("sent_all","mean")).query("n>=30").sort_values("n",ascending=False).round(3).to_dict()
res["genre_agg"] = {
    str(k): {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv) for kk, vv in v.items()}
    for k, v in res["genre_agg"].items()
}

# 折扣有效样本
d = df.dropna(subset=["orig_num","final_num"]).copy()
res["n_discount_rows"] = int(len(d))
res["discount_dist"] = df["discount"].value_counts().head(8).to_dict()

with io.open("diag_report.json","w",encoding="utf-8") as f:
    json.dump(res,f,ensure_ascii=False,indent=2,default=str)
print("ok")
