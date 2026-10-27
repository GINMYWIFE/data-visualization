# -*- coding: utf-8 -*-
"""
price_parser.py —— Steam 西语价格字段解析器 v2

修复的解析缺陷：
  缺陷1: price 字段混入整段促销文案，正则误抽到年份/序号
         'Car Mechanic Simulator 2021 Demo' -> 2021  （原实现，严重错误）
  缺陷2: 西语零价格式 '24,--€' 的 '--' 不被数字正则支持，导致整行价格丢失
  缺陷3: 复合串 '-20%12,--€9,60€' 先抽到原价而非折后价
  缺陷4: 复合串 '-60%9,99€3,99€' 同上 —— 折后价是最后一个金额
"""
import re

# 金额：数字部分 + 可选的西语 '--' 零价尾缀 + 币种符号
AMOUNT_RE = re.compile(
    r"(?<![\d.,])"                                    # 左侧不可紧邻数字/逗号/点
    r"(\d{1,3}(?:[.\s]\d{3})*(?:,\d{1,2})?"           # 1234.56 / 1.234,56 / 19,99
    r"|\d+(?:[.,]\d{1,2})?)"                          # 19,99 / 19.99
    r"(?:,--)?"                                       # 西语零价尾缀：24,--€
    r"\s*(?:€|EUR|EUR\b)"                            # 币种符号（必需，防止误抽年份）
)


def _to_float(raw):
    """西语数字串 -> float。'1.234,56' / '19,99' / '1234.56' / '24,--'"""
    raw = raw.replace(",--", "")
    if "," in raw and "." in raw:
        raw = raw.replace(".", "").replace(",", ".")
    elif "," in raw:
        raw = raw.replace(",", ".")
    elif "." in raw:
        # '1.234'（千分位，无小数）vs '19.99'（小数点）
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", raw):
            raw = raw.replace(".", "")
    try:
        return float(raw)
    except ValueError:
        return None


def parse_amount(text, take="last"):
    """从字符串中提取金额（欧元）。无金额返回 None。

    take='last'  取最后一个金额（用于 '原价€折后价' 这类复合串，取折后价）
    take='first' 取第一个金额（用于 '起订价€续订价' 这类订阅串，取最低门槛价）
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    hits = AMOUNT_RE.findall(s)
    if not hits:
        return None
    raw = hits[-1] if take == "last" else hits[0]
    return _to_float(raw)


# 折扣百分比
PCT_RE = re.compile(r"-\s*(\d{1,3})\s*%")


def parse_pct(text):
    """提取折扣百分比（正数 0-100）。"""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    m = PCT_RE.search(s)
    if not m:
        return None
    v = int(m.group(1))
    return v if 0 <= v <= 100 else None


def parse_count(text):
    """'(3,113)' -> 3113"""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    m = re.search(r"\d[\d.,]*", s)
    if not m:
        return None
    raw = re.sub(r"[^\d]", "", m.group(0))
    return int(raw) if raw else None


# 订阅型商品识别（含按月/按年续订文案）
SUBSCRIPTION_PAT = re.compile(
    r"al mes|meses|renovaci[oó]n|subscription|suscripci[oó]n|"
    r"cada \d+|\d+ por mes|per month|monthly|annual|yearly|"
    r"bienvenida|welcome offer", re.I
)


def is_subscription(text):
    """判断是否为订阅制/长期套餐型商品（其 price 字段是文案而非单一价格）。"""
    if text is None:
        return False
    return bool(SUBSCRIPTION_PAT.search(str(text)))


if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    AMOUNT_TESTS = [
        ("19,99€", 19.99, "last"),
        ("1.234,56 €", 1234.56, "last"),
        ("4,49€", 4.49, "last"),
        ("Car Mechanic Simulator 2021 Demo", None, "last"),
        ("8-Bit Adventures 2 Demo", None, "last"),
        ("24,--€", 24.0, "last"),
        ("0,--€", 0.0, "last"),
        ("-100%   0,--€", 0.0, "last"),
        ("-10%   0,--€", 0.0, "last"),
        ("-20%12,--€9,60€", 9.60, "last"),
        ("-60%9,99€3,99€", 3.99, "last"),
        ("70,--€", 70.0, "last"),
        ("136,27€", 136.27, "last"),
        ("Gratuito", None, "last"),
        ("Free to Play", None, "last"),
        ("A partir de 3,99€ al mes ... 24,99€ cada 12 meses", 3.99, "first"),
    ]
    ok = True
    print("=== parse_amount ===")
    for src, exp, take in AMOUNT_TESTS:
        got = parse_amount(src, take=take)
        good = (exp is None and got is None) or (exp is not None and got is not None and abs(got - exp) < 0.01)
        ok = ok and good
        print(f"  {'OK  ' if good else 'FAIL'} take={take:5} {src!r:50} -> {got}  (expect {exp})")

    print("\n=== parse_pct ===")
    for src, exp in [("-10%", 10), ("-98%", 98), ("136,27€", None), ("-20%", 20), (None, None)]:
        got = parse_pct(src)
        good = got == exp
        ok = ok and good
        print(f"  {'OK  ' if good else 'FAIL'} {src!r:12} -> {got}  (expect {exp})")

    print("\n=== is_subscription ===")
    for src, exp in [("A partir de 3,99€ al mes", True), ("19,99€", False), ("Free to Play", False)]:
        got = is_subscription(src)
        good = got == exp
        ok = ok and good
        print(f"  {'OK  ' if good else 'FAIL'} {src!r:30} -> {got}  (expect {exp})")

    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
