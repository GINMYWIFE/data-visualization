# -*- coding: utf-8 -*-
"""Steam 游戏数据质量体检脚本（数据准备阶段）"""
import pandas as pd
import numpy as np
import re, json, io, sys
from price_parser import parse_amount, parse_pct, parse_count as _pc

CSV = "steam_games.csv"
df = pd.read_csv(CSV, encoding="utf-8", low_memory=False)

out = {}
out["shape"] = df.shape
out["columns"] = list(df.columns)
out["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}

# ---------- 缺失值 ----------
miss = df.isna().sum()
out["missing"] = {c: {"n": int(v), "pct": round(v / len(df) * 100, 2)} for c, v in miss.items() if v > 0}

# ---------- 重复值 ----------
dup_all = int(df.duplicated().sum())
dup_name = int(df.duplicated(subset=["name"]).sum())
out["duplicates"] = {"full_row": dup_all, "by_name": dup_name}

# ---------- 价格解析（西班牙语格式：1.234,56 €） ----------
# 改用 price_parser v2：修复年份误抽取(2,021€伪值)、零价式(24,--€)、复合串取错金额
def parse_price(s):
    v = parse_amount(s, take="last")
    return np.nan if v is None else v

def parse_reviews(s):
    """'(3,113)' -> 3113"""
    if pd.isna(s):
        return np.nan
    m = re.search(r"\d[\d.,]*", str(s))
    if not m:
        return np.nan
    raw = re.sub(r"[^\d]", "", m.group(0))
    return float(raw) if raw else np.nan

df["price_num"] = df["price"].map(parse_price)
df["orig_price_num"] = df["discount_original_price"].map(parse_price)
df["final_price_num"] = df["discount_final_price"].map(parse_price)
df["discount_num"] = df["discount"].map(parse_price)
df["recent_reviews_count_num"] = df["recent_reviews_count"].map(parse_reviews)
df["all_reviews_count_num"] = df["all_reviews_count"].map(parse_reviews)

out["price_stats"] = df["price_num"].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).round(2).to_dict()
out["price_missing"] = int(df["price_num"].isna().sum())
out["final_price_missing"] = int(df["final_price_num"].isna().sum())
out["orig_price_missing"] = int(df["orig_price_num"].isna().sum())

# 折扣一致性：final == orig * (1 - discount/100)
d = df.dropna(subset=["orig_price_num", "final_price_num", "discount_num"])
d = d[(d["orig_price_num"] > 0) & (d["discount_num"] > 0) & (d["discount_num"] < 100)]
calc = (d["orig_price_num"] * (1 - d["discount_num"] / 100)).round(2)
out["discount_consistency"] = {
    "n_checked": int(len(d)),
    "match_rate": round(float((calc - d["final_price_num"].abs()).abs().lt(0.02).mean()) * 100, 2),
}

# 负价格 / 零价格
out["price_anomaly"] = {
    "negative_price": int((df["price_num"] < 0).sum()),
    "zero_price": int((df["price_num"] == 0).sum()),
    "free_final_lt1": int((df["final_price_num"] < 1).sum()),
}

# ---------- 评论区解析 ----------
def parse_review_pct(s):
    if pd.isna(s):
        return np.nan
    s = str(s)
    m = re.search(r"(\d+)\s*%", s)
    if m:
        return float(m.group(1))
    return np.nan

df["recent_pct"] = df["recent_reviews"].map(parse_review_pct)
df["all_pct"] = df["all_reviews"].map(parse_review_pct)
out["review_text_values"] = {
    "recent": df["recent_reviews"].dropna().value_counts().head(10).to_dict(),
    "all": df["all_reviews"].dropna().value_counts().head(10).to_dict(),
}
out["review_pct_missing"] = {
    "recent_pct_unparsed": int(df["recent_pct"].isna().sum() - df["recent_reviews"].isna().sum()),
    "all_pct_unparsed": int(df["all_pct"].isna().sum() - df["all_reviews"].isna().sum()),
}

# 评论数逻辑：recent <= all
rv = df.dropna(subset=["recent_reviews_count_num", "all_reviews_count_num"])
out["review_count_logic"] = {
    "n": int(len(rv)),
    "recent_gt_all": int((rv["recent_reviews_count_num"] > rv["all_reviews_count_num"]).sum()),
}

# reviews_anomally
out["reviews_anomally"] = df["reviews_anomally"].value_counts(dropna=False).to_dict()

# ---------- 类型 / 标签 ----------
out["genre_top20"] = df["genre"].value_counts().head(20).to_dict()
out["n_genres"] = int(df["genre"].nunique())
out["developer_top10"] = df["developer"].value_counts().head(10).to_dict()
out["n_developers"] = int(df["developer"].nunique())
out["n_publishers"] = int(df["publisher"].nunique())
out["n_tags"] = int(df["tags"].notna().sum())
tag_counts = {}
for t in df["tags"].dropna():
    for x in str(t).split(","):
        tag_counts[x.strip()] = tag_counts.get(x.strip(), 0) + 1
out["tags_top20"] = dict(sorted(tag_counts.items(), key=lambda kv: -kv[1])[:20])
out["n_unique_tags"] = len(tag_counts)

# ---------- DLC ----------
out["is_dlc"] = df["is_dlc"].value_counts(dropna=False).to_dict()

# ---------- 发布日期 ----------
def parse_date(s):
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    for fmt in ("%d %b %Y", "%d %B %Y", "%b %d, %Y", "%Y-%m-%d", "%d %b %y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            continue
    return pd.to_datetime(s, errors="coerce")

df["release_dt"] = df["release_date"].map(parse_date)
out["release_date"] = {
    "parsed": int(df["release_dt"].notna().sum()),
    "unparsed": int(df["release_dt"].isna().sum()),
    "min": str(df["release_dt"].min().date()) if df["release_dt"].notna().any() else None,
    "max": str(df["release_dt"].max().date()) if df["release_dt"].notna().any() else None,
    "by_year": df["release_dt"].dt.year.value_counts().sort_index().to_dict(),
}

# ---------- metacritic ----------
out["metacritic"] = {
    "notna": int(df["metacritic_score"].notna().sum()),
    "pct": round(df["metacritic_score"].notna().mean() * 100, 2),
    "stats": df["metacritic_score"].describe().round(2).to_dict(),
}

# ---------- 关联性（初步） ----------
tmp = df.dropna(subset=["price_num", "all_pct", "recent_reviews_count_num", "all_reviews_count_num", "release_dt"])
tmp = tmp[(tmp["all_reviews_count_num"] > 0)]
out["corr"] = {
    "price_vs_allpct": round(float(tmp["price_num"].corr(tmp["all_pct"])), 4),
    "price_vs_recentpct": round(float(tmp["price_num"].corr(tmp["recent_pct"])), 4),
    "reviewcount_vs_allpct": round(float(tmp["all_reviews_count_num"].corr(tmp["all_pct"])), 4),
    "price_vs_allcount": round(float(np.log1p(tmp["price_num"]).corr(np.log1p(tmp["all_reviews_count_num"]))), 4),
    "metacritic_vs_allpct": round(float(df[["metacritic_score","all_pct"]].dropna().corr().iloc[0,1]), 4),
}

# 价格分布直方图分箱（离群点）
p = df["price_num"].dropna()
q1, q3 = p.quantile(.25), p.quantile(.75)
iqr = q3 - q1
out["price_iqr_outliers"] = {
    "q1": float(q1), "q3": float(q3),
    "upper_fence": float(q3 + 1.5 * iqr),
    "n_outlier": int((p > q3 + 1.5 * iqr).sum()),
    "pct_outlier": round(float((p > q3 + 1.5 * iqr).mean() * 100), 2),
}

# 3σ
mu, sd = p.mean(), p.std()
out["price_3sigma"] = {"mean": round(mu,2), "std": round(sd,2), "upper": round(mu+3*sd,2),
                        "n_outlier": int((p > mu + 3*sd).sum())}

# ---------- 文本类字段长度 ----------
out["desc_len"] = df["short_description"].dropna().str.len().describe().round(1).to_dict()
out["empty_strings"] = {
    "short_description": int((df["short_description"].fillna("").str.strip() == "").sum()),
    "name": int((df["name"].fillna("").str.strip() == "").sum()),
    "tags": int((df["tags"].fillna("").str.strip() == "").sum()),
}

# game_content 字段样例（嵌套结构）
sample = df["game_content"].dropna().head(2).tolist()
out["game_content_sample"] = sample

# 内存
out["memory_mb"] = round(df.memory_usage(deep=True).sum() / 1024**2, 2)

with io.open("data_quality_report.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)

print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
