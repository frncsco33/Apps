import os
from datetime import timedelta
import pandas as pd
import numpy as np
from dateutil import parser
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

CSV_PATH = 'Proyecto KFC - Ventas (2).csv'
REPORT_DIR = 'report'

# Utilities ---------------------------------------------------------------
def ensure_dir(path: str = REPORT_DIR):
    os.makedirs(path, exist_ok=True)

def to_float_money(x):
    if pd.isna(x):
        return np.nan
    return float(str(x).replace('$', '').replace(',', '').strip())

def to_float_num(x):
    if pd.isna(x):
        return np.nan
    return float(str(x).replace(',', '').strip())

def to_float_pct(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace('%', '').replace(' ', '').strip()
    return float(s) / 100.0

def fmt_money(v):
    return f"${v:,.2f}"

def fmt_pct(v):
    return f"{100*v:.1f}%"

def week_to_dt(s):
    try:
        dt = parser.parse(str(s), dayfirst=True)
        return dt - timedelta(days=dt.weekday())
    except Exception:
        return pd.NaT

# ------------------------------------------------------------------------
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rename = {
        'WEEK': 'week',
        'SALES': 'sales_raw',
        'ORDERS': 'orders_raw',
        'AOV (SALES)': 'aov_raw',
        'MARKDOWN ALLY': 'markdown_raw',
        'PRO ORDERS': 'pro_orders_raw',
        'Select Store': 'traffic_raw',
        'SS->ATC': 'ss_atc_raw',
        'ATC->PTC': 'atc_ptc_raw',
        'PTC->OP': 'ptc_op_raw',
        'SS->ORDERS': 'ss_orders_raw',
    }
    df = df.rename(columns=rename)

    df['week'] = df['week'].apply(week_to_dt)
    df['sales'] = df['sales_raw'].apply(to_float_money)
    df['orders'] = df['orders_raw'].apply(to_float_num)
    df['aov'] = df['aov_raw'].apply(to_float_money)
    df['markdown'] = df['markdown_raw'].apply(to_float_pct)
    df['pro_orders'] = df['pro_orders_raw'].apply(to_float_pct)
    df['traffic'] = df['traffic_raw'].apply(to_float_num)
    df['ss_atc'] = df['ss_atc_raw'].apply(to_float_pct)
    df['atc_ptc'] = df['atc_ptc_raw'].apply(to_float_pct)
    df['ptc_op'] = df['ptc_op_raw'].apply(to_float_pct)
    df['ss_orders'] = df['ss_orders_raw'].apply(to_float_pct)

    df = df.sort_values('week').dropna(subset=['week']).reset_index(drop=True)

    df['gmv_calc'] = df['orders'] * df['aov']
    df['conversion_final'] = df['ss_orders']
    return df

# ------------------------------------------------------------------------
def log_info(df: pd.DataFrame):
    print(f"[INFO] Filas: {len(df)}, Columnas: {df.shape[1]}")
    print("[INFO] Rango:", df['week'].min().date(), '→', df['week'].max().date())
    print('[INFO] Nulos por columna:\n', df.isna().sum())

# ------------------------------------------------------------------------
def series_with_trend(df: pd.DataFrame, y: str, title: str, out: str):
    ensure_dir(REPORT_DIR)
    X = (df['week'] - df['week'].min()).dt.days.values.reshape(-1, 1)
    y_val = df[y].values
    reg = LinearRegression().fit(X, y_val)
    y_hat = reg.predict(X)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['week'], y=y_val, mode='lines', name=y))
    fig.add_trace(go.Scatter(x=df['week'], y=y_hat, mode='lines', name='Tendencia'))
    fig.update_layout(title=title, xaxis_title='Semana', yaxis_title=y.title())
    fig.write_html(os.path.join(REPORT_DIR, out), include_plotlyjs='cdn', full_html=True)

# ------------------------------------------------------------------------
def funnel_chart(df: pd.DataFrame):
    funnel_df = df.melt(
        id_vars=['week'],
        value_vars=['ss_atc', 'atc_ptc', 'ptc_op', 'ss_orders'],
        var_name='stage', value_name='rate'
    )
    names = {
        'ss_atc': 'SS→ATC',
        'atc_ptc': 'ATC→PTC',
        'ptc_op': 'PTC→OP',
        'ss_orders': 'SS→ORDERS'
    }
    funnel_df['stage'] = funnel_df['stage'].map(names)
    fig = px.line(funnel_df, x='week', y='rate', color='stage',
                  title='Funnel de conversión semanal')
    fig.update_yaxes(tickformat='.1%')
    fig.write_html(os.path.join(REPORT_DIR, 'funnel.html'), include_plotlyjs='cdn', full_html=True)

# ------------------------------------------------------------------------
def scatter_with_trend(df: pd.DataFrame, x: str, y: str, title: str, out: str, pct_x=False):
    fig = px.scatter(df, x=x, y=y, trendline='ols', title=title)
    if pct_x:
        fig.update_xaxes(tickformat='.1%')
    fig.write_html(os.path.join(REPORT_DIR, out), include_plotlyjs='cdn', full_html=True)

# ------------------------------------------------------------------------
def projection(df: pd.DataFrame):
    last_date = df['week'].max()
    weeks_fwd = pd.date_range(last_date + timedelta(days=7), periods=9, freq='W-MON')

    traffic_last = df['traffic'].iloc[-1]
    conv_last = df['conversion_final'].iloc[-1]
    aov_const = df['aov'].tail(4).mean()

    g_traf_week = (1 + 0.10) ** (1/9) - 1
    g_conv_week = (1 + 0.05) ** (1/9) - 1

    rows = []
    t, c = traffic_last, conv_last
    for w in weeks_fwd:
        t *= (1 + g_traf_week)
        c *= (1 + g_conv_week)
        o = t * c
        s = o * aov_const
        rows.append({'week': w, 'traffic': t, 'conversion_final': c,
                     'orders': o, 'sales': s})
    df_proj = pd.DataFrame(rows)

    fig_o = go.Figure()
    fig_o.add_trace(go.Scatter(x=df['week'], y=df['orders'], mode='lines', name='Histórico'))
    fig_o.add_trace(go.Scatter(x=df_proj['week'], y=df_proj['orders'], mode='lines', name='Proyección 9w'))
    fig_o.update_layout(title='Órdenes: histórico vs proyección')
    fig_o.write_html(os.path.join(REPORT_DIR, 'orders_projection.html'), include_plotlyjs='cdn', full_html=True)

    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=df['week'], y=df['sales'], mode='lines', name='Histórico'))
    fig_s.add_trace(go.Scatter(x=df_proj['week'], y=df_proj['sales'], mode='lines', name='Proyección 9w'))
    fig_s.update_layout(title='Ventas: histórico vs proyección')
    fig_s.write_html(os.path.join(REPORT_DIR, 'sales_projection.html'), include_plotlyjs='cdn', full_html=True)

    inc_o = df_proj['orders'].iloc[-1] - df['orders'].iloc[-1]
    inc_s = df_proj['sales'].iloc[-1] - df['sales'].iloc[-1]
    return df_proj, inc_o, inc_s, aov_const

# ------------------------------------------------------------------------
def render_index(df: pd.DataFrame, df_proj: pd.DataFrame, inc_o: float, inc_s: float):
    ensure_dir(REPORT_DIR)
    last_8 = df.tail(8)
    summary_html = ''.join(
        f"<tr><td>{w.strftime('%Y-%m-%d')}</td>"
        f"<td>{fmt_money(s)}</td>"
        f"<td>{o:,.0f}</td>"
        f"<td>{t:,.0f}</td>"
        f"<td>{fmt_pct(c)}</td>"
        f"<td>{fmt_pct(m)}</td>"
        f"<td>{fmt_money(a)}</td></tr>"
        for w, s, o, t, c, m, a in zip(
            last_8['week'], last_8['sales'], last_8['orders'],
            last_8['traffic'], last_8['conversion_final'],
            last_8['markdown'], last_8['aov']
        )
    )
    html = f"""
<html>
<head>
  <meta charset='utf-8'/>
  <title>Dashboard KFC</title>
  <style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
    h1, h2 {{ margin: 8px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
    .card {{ border:1px solid #eee; border-radius:10px; padding:14px; }}
    table {{ border-collapse: collapse; width:100%; }}
    th, td {{ text-align:right; padding:6px 8px; border-bottom:1px solid #eee; }}
    th:first-child, td:first-child {{ text-align:left; }}
  </style>
</head>
<body>
<h1>Dashboard KFC – Ventas, Funnel y Proyección</h1>
<div class='grid'>
  <div class='card'>
    <h2>Series históricas</h2>
    <div><a href='sales_series.html'>Ventas con tendencia</a></div>
    <div><a href='orders_series.html'>Órdenes con tendencia</a></div>
    <div><a href='funnel.html'>Funnel semanal</a></div>
    <div><a href='scatter_traffic_orders.html'>Tráfico vs Órdenes</a></div>
    <div><a href='scatter_markdown_sales.html'>Markdown vs Ventas</a> | <a href='scatter_markdown_orders.html'>Markdown vs Órdenes</a></div>
  </div>
  <div class='card'>
    <h2>Proyección 9 semanas</h2>
    <div><a href='orders_projection.html'>Órdenes: histórico vs proyección</a></div>
    <div><a href='sales_projection.html'>Ventas: histórico vs proyección</a></div>
    <p><b>Incremento estimado al final de 9 semanas</b><br>
       Órdenes: {inc_o:,.0f}<br>
       Ventas: {fmt_money(inc_s)}
    </p>
    <small>Supuestos: +10% tráfico total y +5% conversión total (compuesto), AOV constante (promedio últimas 4 semanas).</small>
  </div>
  <div class='card'>
    <h2>Últimas 8 semanas</h2>
    <table>
      <tr><th>Week</th><th>Sales</th><th>Orders</th><th>Traffic</th><th>SS→ORDERS</th><th>Markdown</th><th>AOV</th></tr>
      {summary_html}
    </table>
  </div>
</div>
</body>
</html>
"""
    with open(os.path.join(REPORT_DIR, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html)

# ------------------------------------------------------------------------
def main():
    df = load_and_clean(CSV_PATH)
    log_info(df)
    ensure_dir(REPORT_DIR)
    series_with_trend(df, 'sales', 'Ventas semanales con tendencia', 'sales_series.html')
    series_with_trend(df, 'orders', 'Órdenes semanales con tendencia', 'orders_series.html')
    funnel_chart(df)
    scatter_with_trend(df, 'traffic', 'orders', 'Tráfico vs Órdenes', 'scatter_traffic_orders.html')
    scatter_with_trend(df, 'markdown', 'sales', 'Markdown vs Ventas', 'scatter_markdown_sales.html', pct_x=True)
    scatter_with_trend(df, 'markdown', 'orders', 'Markdown vs Órdenes', 'scatter_markdown_orders.html', pct_x=True)
    df_proj, inc_o, inc_s, aov_const = projection(df)
    render_index(df, df_proj, inc_o, inc_s)
    print('[OK] Dashboard generado en ./report/index.html')

if __name__ == '__main__':
    main()
