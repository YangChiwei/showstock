import os
from datetime import datetime, timedelta
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mplfinance.original_flavor as mpf
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# -----------------------------------------------------------------------------
# 1. 頁面基本配置
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="台股六大技術指標與AI診斷儀表板",
    page_icon="📈",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. 使用 st.secrets 進行身分驗證 (Authentication)
# -----------------------------------------------------------------------------
def check_login():
    """使用 st.secrets 驗證帳號密碼並控管登入狀態"""
    # 檢查 secrets.toml 是否已正確設定 passwords
    if "passwords" not in st.secrets:
        st.error("⚠️ 未在 `st.secrets` 中找到 `[passwords]` 設定，請先建立 `.streamlit/secrets.toml`。")
        st.stop()

    valid_credentials = st.secrets["passwords"]

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        col1, col2, col3 = st.columns([1, 1.2, 1])
        with col2:
            st.title("🔒 系統登入")
            st.markdown("請輸入使用者名稱與密碼以存取技術指標看板。")
            with st.form("login_form"):
                username = st.text_input("使用者名稱 (Username)")
                password = st.text_input("密碼 (Password)", type="password")
                submit_button = st.form_submit_button("登入系統", use_container_width=True)

                if submit_button:
                    # 比對 secrets 中的帳號密碼
                    if username in valid_credentials and valid_credentials[username] == password:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success("登入成功！正在載入...")
                        st.rerun()
                    else:
                        st.error("❌ 帳號或密碼錯誤，請重新輸入。")
        return False
    return True

# 若未通過登入驗證，立即中斷後續執行
if not check_login():
    st.stop()

# -----------------------------------------------------------------------------
# 3. 系統字型配置
# -----------------------------------------------------------------------------
font_path = "NotoSansTC-Regular.ttf"
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    st.sidebar.warning(f"⚠️ 未在同目錄找到 `{font_path}`，使用系統預設字型。")
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans', 'sans-serif']

plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------------------
# 4. 輔助計算函式
# -----------------------------------------------------------------------------
def calculate_yahoo_rsi(series: pd.Series, period: int) -> pd.Series:[cite: 1]
    """計算 Yahoo Finance 相同邏輯的 RSI (Wilder's Smoothing)"""
    delta = series.diff()[cite: 1]
    gain = delta.clip(lower=0)[cite: 1]
    loss = (-delta).clip(lower=0)[cite: 1]
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()[cite: 1]
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()[cite: 1]
    
    rs = avg_gain / avg_loss[cite: 1]
    return 100 - (100 / (1 + rs))[cite: 1]

@st.cache_data(ttl=3600)
def load_and_calculate_data(
    stock_id: str,
    days: int,
    ma_short: int, ma_mid: int, ma_long: int,
    bb_period: int, bb_std: float,
    kd_period: int,
    macd_fast: int, macd_slow: int, macd_signal: int,
    rsi_short: int, rsi_long: int,
    bias_short: int, bias_long: int
):
    """下載數據並動態計算技術指標"""
    end_date = datetime.today().date()[cite: 1]
    start = end_date - timedelta(days=days)
    warmup_days = max(60, ma_long * 2, macd_slow * 2, bb_period * 2)
    start_date = start - timedelta(days=warmup_days)
    
    df = yf.download(stock_id, start=start_date, end=end_date)[cite: 1]
    if df.empty:
        return None
    
    if isinstance(df.columns, pd.MultiIndex):[cite: 1]
        df.columns = df.columns.get_level_values(0)[cite: 1]
        
    # 1. 均線與布林通道
    df[f'SMA_{ma_short}'] = df['Close'].rolling(window=ma_short).mean()
    df[f'SMA_{ma_mid}'] = df['Close'].rolling(window=ma_mid).mean()
    df[f'SMA_{ma_long}'] = df['Close'].rolling(window=ma_long).mean()
    
    df['bb_mid'] = df['Close'].rolling(window=bb_period).mean()
    df['bb_std'] = df['Close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * bb_std)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * bb_std)

    # 2. KD / J 值
    low_min = df['Low'].rolling(window=kd_period).min()
    high_max = df['High'].rolling(window=kd_period).max()
    df['RSV'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
    df['K'] = df['RSV'].ewm(alpha=1/3, adjust=False).mean()[cite: 1]
    df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()[cite: 1]
    df['J'] = 3 * df['D'] - 2 * df['K'][cite: 1]

    # 3. OBV 能量潮
    df['OBV'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], -df['Volume'])[cite: 1]
    df['OBV'] = df['OBV'].cumsum()[cite: 1]
    df['OBV_MA'] = df['OBV'].rolling(10).mean()

    # 4. MACD
    df['EMA_fast'] = df['Close'].ewm(span=macd_fast, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=macd_slow, adjust=False).mean()
    df['DIF'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD'] = df['DIF'].ewm(span=macd_signal, adjust=False).mean()
    df['MACD Histogram'] = df['DIF'] - df['MACD'][cite: 1]

    # 5. RSI
    df[f'RSI_{rsi_short}'] = calculate_yahoo_rsi(df['Close'], period=rsi_short)
    df[f'RSI_{rsi_long}'] = calculate_yahoo_rsi(df['Close'], period=rsi_long)

    # 6. BIAS 乖離率
    df[f'BIAS_{bias_short}'] = ((df['Close'] - df['Close'].rolling(bias_short).mean()) / df['Close'].rolling(bias_short).mean()) * 100
    df[f'BIAS_{bias_long}'] = ((df['Close'] - df['Close'].rolling(bias_long).mean()) / df['Close'].rolling(bias_long).mean()) * 100
    df['BIAS_DIFF'] = df[f'BIAS_{bias_short}'] - df[f'BIAS_{bias_long}']

    df = df.loc[start:, :].copy()[cite: 1]
    df.index = df.index.map(lambda x: x.strftime('%y-%m-%d'))[cite: 1]
    return df

# -----------------------------------------------------------------------------
# 5. 六大指標動態診斷邏輯
# -----------------------------------------------------------------------------
def analyze_indicators(df: pd.DataFrame, params: dict):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    analysis = []
    bullish_cnt = 0
    bearish_cnt = 0

    # 1. 均線與布林通道
    close, bb_mid, up, low = last['Close'], last['bb_mid'], last['bb_upper'], last['bb_lower']
    if close > up:
        bb_status = f"🔴 突破布林上軌 ({params['bb_std']}σ 壓力區)"
        bb_sig = "偏多但防拉回"
        bullish_cnt += 1
    elif close < low:
        bb_status = f"🟢 跌破布林下軌 ({params['bb_std']}σ 超跌區)"
        bb_sig = "偏空但防反彈"
        bearish_cnt += 1
    elif close > bb_mid:
        bb_status = f"🔴 站上中軌 ({params['bb_period']}MA 多頭支撐)"
        bb_sig = "多方"
        bullish_cnt += 1
    else:
        bb_status = f"🟢 跌破中軌 ({params['bb_period']}MA 空頭壓制)"
        bb_sig = "空方"
        bearish_cnt += 1
    analysis.append({
        "指標項目": f"1. 均線與布林帶 ({params['bb_period']}日, {params['bb_std']}σ)",
        "最新數值": f"收盤: {close:.2f} | 中軌: {bb_mid:.2f} [下軌: {low:.2f} ~ 上軌: {up:.2f}]",
        "狀態診斷": bb_status,
        "多空判定": bb_sig
    })

    # 2. OBV 能量潮
    obv, obv_ma = last['OBV'], last['OBV_MA']
    if obv > obv_ma and last['OBV'] > prev['OBV']:
        obv_status = "🔴 量能持續匯聚，突破 10 日能量均線"
        obv_sig = "多方"
        bullish_cnt += 1
    else:
        obv_status = "🟢 資金動能趨緩，跌破 10 日能量均線"
        obv_sig = "空方"
        bearish_cnt += 1
    analysis.append({
        "指標項目": "2. OBV 能量潮",
        "最新數值": f"OBV: {obv:,.0f} (10MA: {obv_ma:,.0f})",
        "狀態診斷": obv_status,
        "多空判定": obv_sig
    })

    # 3. KD / J 值
    k, d, j = last['K'], last['D'], last['J']
    pk, pd_val = prev['K'], prev['D']
    if pk <= pd_val and k > d:
        kd_status = "🔴 黃金交叉 (買進訊號)"
        kd_sig = "強烈多方"
        bullish_cnt += 1
    elif pk >= pd_val and k < d:
        kd_status = "🟢 死亡交叉 (賣出訊號)"
        kd_sig = "強烈空方"
        bearish_cnt += 1
    elif k > d:
        kd_status = "🔴 K > D 多方續抱 (留意高檔鈍化)"
        kd_sig = "多方"
        bullish_cnt += 1
    else:
        kd_status = "🟢 K < D 空方觀望"
        kd_sig = "空方"
        bearish_cnt += 1
    analysis.append({
        "指標項目": f"3. KD / J 指標 (N={params['kd_period']})",
        "最新數值": f"K: {k:.2f} | D: {d:.2f} | J: {j:.2f}",
        "狀態診斷": kd_status,
        "多空判定": kd_sig
    })

    # 4. MACD
    dif, macd, hist = last['DIF'], last['MACD'], last['MACD Histogram']
    if hist > 0 and hist > prev['MACD Histogram']:
        macd_status = "🔴 紅柱擴大 (多方動能增強)"
        macd_sig = "多方"
        bullish_cnt += 1
    elif hist > 0 and hist <= prev['MACD Histogram']:
        macd_status = "🟡 紅柱收縮 (多方動能趨緩)"
        macd_sig = "中性偏多"
    elif hist < 0 and hist < prev['MACD Histogram']:
        macd_status = "🟢 綠柱擴大 (空方壓力增強)"
        macd_sig = "空方"
        bearish_cnt += 1
    else:
        macd_status = "🟡 綠柱收縮 (空方壓力趨緩)"
        macd_sig = "中性偏空"
    analysis.append({
        "指標項目": f"4. MACD ({params['macd_fast']}, {params['macd_slow']}, {params['macd_signal']})",
        "最新數值": f"DIF: {dif:.2f} | MACD: {macd:.2f} | 柱狀體: {hist:.2f}",
        "狀態診斷": macd_status,
        "多空判定": macd_sig
    })

    # 5. RSI
    rsi_s = last[f"RSI_{params['rsi_short']}"]
    rsi_l = last[f"RSI_{params['rsi_long']}"]
    if rsi_s >= 80:
        rsi_status = f"⚠️ RSI{params['rsi_short']} >= 80 (嚴重超買，留意過熱回檔)"
        rsi_sig = "過熱警戒"
    elif rsi_s <= 20:
        rsi_status = f"✨ RSI{params['rsi_short']} <= 20 (嚴重超賣，反彈契機)"
        rsi_sig = "超跌契機"
    elif rsi_s > rsi_l:
        rsi_status = f"🔴 短期 RSI{params['rsi_short']} > 長期 RSI{params['rsi_long']} (向上推升)"
        rsi_sig = "多方"
        bullish_cnt += 1
    else:
        rsi_status = f"🟢 短期 RSI{params['rsi_short']} < 長期 RSI{params['rsi_long']} (向下修正)"
        rsi_sig = "空方"
        bearish_cnt += 1
    analysis.append({
        "指標項目": f"5. RSI 相對強弱 ({params['rsi_short']}日 / {params['rsi_long']}日)",
        "最新數值": f"RSI({params['rsi_short']}): {rsi_s:.2f} | RSI({params['rsi_long']}): {rsi_l:.2f}",
        "狀態診斷": rsi_status,
        "多空判定": rsi_sig
    })

    # 6. BIAS 乖離率
    bias_s = last[f"BIAS_{params['bias_short']}"]
    bias_l = last[f"BIAS_{params['bias_long']}"]
    diff_bias = last['BIAS_DIFF']
    if diff_bias > 0:
        bias_status = f"🔴 (B{params['bias_short']}-B{params['bias_long']}) 正乖離放大，短線力道強於中線"
        bias_sig = "多方"
        bullish_cnt += 1
    else:
        bias_status = f"🟢 (B{params['bias_short']}-B{params['bias_long']}) 負乖離加深，短線力道弱於中線"
        bias_sig = "空方"
        bearish_cnt += 1
    analysis.append({
        "指標項目": f"6. BIAS 乖離率 ({params['bias_short']}日 / {params['bias_long']}日)",
        "最新數值": f"BIAS({params['bias_short']}): {bias_s:.2f}% | BIAS({params['bias_long']}): {bias_l:.2f}% | 差值: {diff_bias:.2f}%",
        "狀態診斷": bias_status,
        "多空判定": bias_sig
    })

    return pd.DataFrame(analysis), bullish_cnt, bearish_cnt, last

# -----------------------------------------------------------------------------
# 6. 側邊欄控制項 (帳號管理、股票選擇與指標參數)
# -----------------------------------------------------------------------------
st.sidebar.markdown(f"👤 **目前登入者**: `{st.session_state.username}`")
if st.sidebar.button("🚪 登出系統", use_container_width=True):
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🎯 標的與時間設定")

popular_stocks = {
    "台積電 (2330.TW)": "2330.TW",
    "聯發科 (2454.TW)": "2454.TW",
    "鴻海 (2317.TW)": "2317.TW",
    "廣達 (2382.TW)": "2382.TW",
    "台達電 (2308.TW)": "2308.TW",
    "元大台灣50 (0050.TW)": "0050.TW",
    "元大高股息 (0056.TW)": "0056.TW",
    "國泰永續高股息 (00878.TW)": "00878.TW",
    "富邦台50 (006208.TW)": "006208.TW",
    "輝達 NVIDIA (NVDA)": "NVDA",
    "蘋果 Apple (AAPL)": "AAPL",
    "微軟 Microsoft (MSFT)": "MSFT",
    "自訂輸入代碼...": "CUSTOM"
}

stock_selection = st.sidebar.selectbox("選擇預設標的", list(popular_stocks.keys()), index=0)

if popular_stocks[stock_selection] == "CUSTOM":
    stock_id = st.sidebar.text_input("輸入股票代碼 (台股請加 .TW 或 .TWO)", value="2330.TW")
else:
    stock_id = popular_stocks[stock_selection]

days = st.sidebar.slider("查詢天數 (天)", min_value=30, max_value=365, value=180, step=10)

st.sidebar.markdown("---")
st.sidebar.header("🛠️ 六大指標參數設定")

with st.sidebar.expander("1. 均線與布林帶參數", expanded=False):
    ma_short = st.number_input("短均線 (天)", value=5, min_value=1, max_value=60)
    ma_mid = st.number_input("中均線 (天)", value=10, min_value=2, max_value=120)
    ma_long = st.number_input("長均線 (天)", value=20, min_value=5, max_value=240)
    bb_period = st.number_input("布林帶週期 (天)", value=20, min_value=5, max_value=100)
    bb_std = st.number_input("布林帶標準差倍數 (σ)", value=2.0, min_value=1.0, max_value=3.5, step=0.1)

with st.sidebar.expander("2. KD 指標參數", expanded=False):
    kd_period = st.number_input("RSV 計算週期 (天)", value=9, min_value=3, max_value=60)

with st.sidebar.expander("3. MACD 指標參數", expanded=False):
    macd_fast = st.number_input("快線 EMA (天)", value=12, min_value=2, max_value=60)
    macd_slow = st.number_input("慢線 EMA (天)", value=26, min_value=5, max_value=120)
    macd_signal = st.number_input("訊號線 Signal (天)", value=9, min_value=2, max_value=30)

with st.sidebar.expander("4. RSI 指標參數", expanded=False):
    rsi_short = st.number_input("短週期 RSI (天)", value=5, min_value=2, max_value=30)
    rsi_long = st.number_input("長週期 RSI (天)", value=10, min_value=5, max_value=60)

with st.sidebar.expander("5. BIAS 乖離率參數", expanded=False):
    bias_short = st.number_input("短期乖離 (天)", value=10, min_value=2, max_value=60)
    bias_long = st.number_input("長期乖離 (天)", value=20, min_value=5, max_value=120)

params = {
    "ma_short": ma_short, "ma_mid": ma_mid, "ma_long": ma_long,
    "bb_period": bb_period, "bb_std": bb_std,
    "kd_period": kd_period,
    "macd_fast": macd_fast, "macd_slow": macd_slow, "macd_signal": macd_signal,
    "rsi_short": rsi_short, "rsi_long": rsi_long,
    "bias_short": bias_short, "bias_long": bias_long
}

# -----------------------------------------------------------------------------
# 7. 主畫面呈現
# -----------------------------------------------------------------------------
st.title(f"📊 {stock_id} 股市技術分析看板")

with st.spinner(f"正在載入 {stock_id} 數據並繪製圖表..."):
    df = load_and_calculate_data(stock_id, days, **params)

if df is None or len(df) == 0:
    st.error(f"查無 `{stock_id}` 之股票資料，請檢查代碼是否正確。")
else:
    analysis_df, bull_score, bear_score, latest_row = analyze_indicators(df, params)

    # 頂部數據卡片
    col1, col2, col3, col4 = st.columns(4)
    latest_date = df.index[-1]
    latest_close = latest_row['Close']
    prev_close = df.iloc[-2]['Close']
    change = latest_close - prev_close
    pct_change = (change / prev_close) * 100

    col1.metric("最新交易日", latest_date)
    col1.metric("收盤價", f"{latest_close:.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
    col2.metric("多方指標數", f"{bull_score} / 6", delta="偏多格局" if bull_score > bear_score else None)
    col3.metric("空方指標數", f"{bear_score} / 6", delta="-偏空格局" if bear_score > bull_score else None, delta_color="inverse")
    
    if bull_score >= 4:
        summary_text = "🟢 **強烈偏多格局**：多數自訂指標呈正向或攻擊訊號，建議順勢操作或持股續抱。"
    elif bear_score >= 4:
        summary_text = "🔴 **空方壓力偏重**：多數指標轉弱或死叉，建議做好資金風險控管。"
    else:
        summary_text = "🟡 **多空震盪整理**：多空訊號互現，建議逢支撐低接或等待方向明朗。"
    col4.info(f"**綜合診斷結論**\n\n{summary_text}")

    st.markdown("---")

    tab1, tab2 = st.tabs(["📈 完整技術圖表看板", "📝 六大指標詳細診斷"])

    with tab1:
        st.subheader("📉 六大指標同步走勢圖")
        x_ticks_pos = range(0, len(df.index), max(1, len(df.index) // 10))
        x_ticks_labels = df.index[x_ticks_pos]

        fig = plt.figure(figsize=(14, 18), layout='constrained')

        # 1. K線與布林帶 (1~3格)
        ax1 = fig.add_subplot(8, 1, (1, 3))[cite: 1]
        ax1.set_xticks(x_ticks_pos)[cite: 1]
        ax1.set_xticklabels(x_ticks_labels)[cite: 1]
        mpf.candlestick2_ochl(ax1, df['Open'], df['Close'], df['High'], df['Low'], width=0.8, colorup='r', colordown='g', alpha=1)[cite: 1]
        ax1.plot(df[f'SMA_{ma_short}'], label=f'{ma_short}日均線', alpha=0.9, color='cyan', lw=0.8)
        ax1.plot(df[f'SMA_{ma_mid}'], label=f'{ma_mid}日均線', alpha=0.9, color='purple', lw=0.8)
        ax1.plot(df[f'SMA_{ma_long}'], label=f'{ma_long}日均線', alpha=0.9, color='orange', lw=0.8)
        ax1.plot(df['bb_upper'], label=f'布林上軌({bb_std}σ)', alpha=0.9, color='g', ls=':')
        ax1.plot(df['bb_lower'], label=f'布林下軌({bb_std}σ)', alpha=0.9, color='g', ls=':')
        ax1.legend(loc="upper left")
        ax1.set_title(f"{stock_id} K線與指標全貌 ({df.index[0]} ~ {df.index[-1]})", fontsize=14)

        # 2. OBV 與 交易量 (第 4 格)
        ax2 = fig.add_subplot(8, 1, 4)[cite: 1]
        ax2.set_xticks(x_ticks_pos)[cite: 1]
        ax2.set_xticklabels([])[cite: 1]
        vol_colors = np.where(df['Close'] >= df['Close'].shift(1), 'r', 'g')
        ax2.plot(df['OBV'], color='purple', linestyle='--', label='OBV')[cite: 1]
        ax2.legend(loc="upper left")
        
        ax2_1 = ax2.twinx()[cite: 1]
        ax2_1.bar(df.index, height=df['Volume'], color=vol_colors, width=0.8, alpha=0.5)
        red_patch = mpatches.Patch(color='red', label='紅漲')
        green_patch = mpatches.Patch(color='green', label='綠跌')
        ax2_1.legend(handles=[red_patch, green_patch], loc="upper right", title="交易量")

        # 3. KD 與 J 值 (第 5 格)
        ax3 = fig.add_subplot(8, 1, 5)[cite: 1]
        ax3.plot(df['K'], label=f'K 線 ({kd_period})', color='cyan', lw=0.8)
        ax3.plot(df['D'], label='D 線', color='purple', lw=0.8)
        ax3.plot(df['J'], label='J 線', linestyle='--', color='orange')
        ax3.set_xticks(x_ticks_pos)[cite: 1]
        ax3.set_xticklabels([])
        ax3.legend(loc="upper left")

        # 4. MACD (第 6 格)
        ax4 = fig.add_subplot(8, 1, 6)[cite: 1]
        ax4.plot(df['DIF'], label='DIF', color='purple')
        ax4.plot(df['MACD'], label='MACD', color='skyblue')[cite: 1]
        macd_colors = np.where(df['MACD Histogram'] >= 0, 'r', 'g')[cite: 1]
        ax4.bar(df.index, height=df['MACD Histogram'], color=macd_colors, alpha=0.8)[cite: 1]
        ax4.axhline(0, color='gray', linestyle='--', linewidth=1.0)
        ax4.set_xticks(x_ticks_pos)[cite: 1]
        ax4.set_xticklabels([])[cite: 1]
        macd_red_patch = mpatches.Patch(color='red', label='MACD多頭(紅)')
        macd_green_patch = mpatches.Patch(color='green', label='MACD空頭(綠)')
        handles, labels = ax4.get_legend_handles_labels()[cite: 1]
        handles.extend([macd_red_patch, macd_green_patch])[cite: 1]
        ax4.legend(handles=handles, loc="upper left", fontsize=8)

        # 5. RSI (第 7 格)
        ax5 = fig.add_subplot(8, 1, 7)[cite: 1]
        ax5.plot(df[f'RSI_{rsi_short}'], label=f'RSI {rsi_short}', color='cyan', lw=0.8)
        ax5.plot(df[f'RSI_{rsi_long}'], label=f'RSI {rsi_long}', color='purple', lw=0.8)
        ax5.set_xticks(x_ticks_pos)[cite: 1]
        ax5.set_xticklabels([])[cite: 1]
        ax5.set_ylim(0, 100)[cite: 1]
        ax5.axhline(70, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        ax5.axhline(30, color='green', linestyle='--', linewidth=0.8, alpha=0.6)
        ax5.legend(loc="upper left")

        # 6. BIAS 乖離率 (第 8 格)
        ax6 = fig.add_subplot(8, 1, 8)[cite: 1]
        ax6.plot(df[f'BIAS_{bias_short}'], label=f'BIAS {bias_short}', color='cyan', lw=0.8)
        ax6.plot(df[f'BIAS_{bias_long}'], label=f'BIAS {bias_long}', color='purple', lw=0.8)
        bias_colors = np.where(df['BIAS_DIFF'] >= 0, 'r', 'g')
        ax6.bar(df.index, height=df['BIAS_DIFF'], color=bias_colors, alpha=0.8)
        ax6.axhline(0, color='gray', linestyle='--', linewidth=1.0)
        bias_red_patch = mpatches.Patch(color='red', label=f'B{bias_short}-B{bias_long} 正強')
        bias_green_patch = mpatches.Patch(color='green', label=f'B{bias_short}-B{bias_long} 負弱')
        handles, labels = ax6.get_legend_handles_labels()[cite: 1]
        handles.extend([bias_red_patch, bias_green_patch])[cite: 1]
        ax6.set_xticks(x_ticks_pos)[cite: 1]
        ax6.set_xticklabels(x_ticks_labels, rotation=45)
        ax6.legend(handles=handles, loc="upper left", fontsize=8)

        st.pyplot(fig)

    with tab2:
        st.subheader("📋 最新指標數值與狀態診斷")
        st.table(analysis_df)

        st.markdown(f"""
        #### ⚙️ 目前套用之自訂參數：
        - **均線**: {ma_short}日 / {ma_mid}日 / {ma_long}日
        - **布林帶**: 週期 {bb_period}日 | 標準差 {bb_std}σ
        - **KD**: RSV週期 {kd_period}日
        - **MACD**: 快線 {macd_fast} | 慢線 {macd_slow} | 訊號線 {macd_signal}
        - **RSI**: 短期 {rsi_short}日 | 長期 {rsi_long}日
        - **BIAS**: 短期 {bias_short}日 | 長期 {bias_long}日
        """)