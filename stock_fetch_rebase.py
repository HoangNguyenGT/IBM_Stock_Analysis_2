import yfinance as yf
import pandas as pd

TICKERS = {
    "IBM":  "IBM",          # IBM common stock
    "SP500": "^GSPC",       # S&P 500 index
    "GOLD": "GC=F",         # COMEX Gold futures
    "BOND": "IEF",          # 7-10 yr Treasury ETF  (alt: "TLT" or "^TNX")
}
START_DATE = "2002-07-26"
CACHE_CSV  = "market_data_rebased.csv"
MAX_RETRIES = 2             # one retry if Yahoo sends back an empty frame

def safe_dl(tkr, start, tries=MAX_RETRIES):
    for _ in range(tries):
        df = yf.download(tkr, start=start,
                         auto_adjust=True, progress=False)["Close"]
        if not df.empty:
            df.name = tkr          # <-- give the Series a column name
            return df
    raise RuntimeError(f"No data returned for {tkr}")

# 1 – fetch each asset separately
frames = {}
for code in TICKERS.values():
    print(f"⇢ downloading {code}")
    frames[code] = safe_dl(code, START_DATE)

# 2 – merge on the *intersection* of available dates
raw = pd.concat(frames, axis=1, join="inner")
raw.columns = [k for k in TICKERS]         # friendly order: IBM, SP500…

# 3 – forward-fill 1–2-day holiday gaps
raw = raw.ffill()

# 4 – re-base SP500, GOLD, BOND to IBM’s first price
for col in ["SP500", "GOLD", "BOND"]:
    raw[col] *= raw["IBM"].iloc[0] / raw[col].iloc[0]

raw.to_csv(CACHE_CSV)
print("✅ saved", CACHE_CSV, "\n", raw.head())
