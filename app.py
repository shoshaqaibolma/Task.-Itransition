import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import STORES, solve
from llm import chat, load_config
from tools import TABLES, VIEWS, _table, default_state, table_view

st.set_page_config(page_title="Bookstore Analytics", page_icon="📚", layout="wide")

EXAMPLES = [
    "What were the top 5 days by revenue in STORE2?",
]


@st.cache_data(show_spinner="Crunching the store data…")
def results(store):
    return solve(store)



if "dash" not in st.session_state:
    st.session_state.dash = default_state()
if "messages" not in st.session_state:
    st.session_state.messages = []
dash = st.session_state.dash


def on_store_change():
    if st.session_state.store_widget:
        dash["store"] = st.session_state.store_widget


def on_view_change():
    dash["view"] = st.session_state.view_widget


config = load_config(st.secrets)
with st.sidebar:
    st.header("💬 Ask the data")
    if config.get("api_key"):
        st.caption(f"Model: {config['model']} · answers come from exact pandas tools, not from the model's memory")
    else:
        st.warning("AI is not configured. Add LLM_API_KEY to the app secrets.")

    history_box = st.container(height=520, border=False)   # input stays visible below a scrolling history
    question = st.chat_input("Ask a question or give a command…")

    with st.expander("Try asking"):
        for i, ex in enumerate(EXAMPLES):
            if st.button(ex, key=f"ex{i}", width="stretch"):
                question = ex

    if st.session_state.messages and st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    if question:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": question})
        with history_box, st.spinner("Thinking…"):
            try:
                answer, trace = chat(question, history, dash, config)
            except Exception as e:  # noqa: BLE001 - show API problems in the chat, keep the app alive
                answer, trace = f"⚠️ The AI service returned an error: {e}", []
        st.session_state.messages.append({"role": "assistant", "content": answer, "trace": trace})

    with history_box:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("trace"):
                    with st.expander(f"🔧 {len(m['trace'])} tool call(s)"):
                        for t in m["trace"]:
                            st.markdown(f"**{t['tool']}**")
                            st.json(t["args"], expanded=False)
                            st.caption(t["result"])

st.session_state.store_widget = dash["store"]
st.session_state.view_widget = dash["view"]


st.title(" Bookstore Analytics")
st.segmented_control("Store", STORES, key="store_widget", on_change=on_store_change)
store = dash["store"]
r = results(store)

tabs = st.tabs(VIEWS, key="view_widget", on_change=on_view_change)
tab = dict(zip(VIEWS, tabs))

with tab["Overview"]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unique users", f"{r['unique_users']:,}", border=True)
    c2.metric("Unique author sets", f"{r['author_sets']:,}", border=True)
    c3.metric("Total revenue", f"${r['total_revenue']:,.2f}", border=True)
    c4.metric("Orders", f"{r['orders']:,}", border=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Top 5 days by revenue")
        st.dataframe(
            pd.DataFrame({"Date": r["top5_days"], "Revenue": r["top5_revenue"]}),
            column_config={"Revenue": st.column_config.NumberColumn(format="$%,.2f")},
            hide_index=True, width="stretch",
        )
    with right:
        st.subheader("Most popular author(s)")
        st.markdown("**" + ", ".join(r["top_authors"]) + "**")
        st.caption(f"{r['top_authors_sold']} books sold")
        st.subheader("Best buyer")
        st.code(str(r["best_buyer_ids"]), language=None)
        st.caption(f"${r['best_buyer_spend']:,.2f} spent · all ids of the same person (aliases)")

    st.subheader("Daily revenue")
    df = r["daily"].rename("Revenue").reset_index()
    fig = px.line(df, x="date", y="Revenue", labels={"date": "", "Revenue": "USD"})
    fig.update_traces(line_width=1)
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch", key="overview_chart")

with tab["Revenue"]:
    ch = dash["chart"]
    series = r["daily"]
    if ch.get("date_from"):
        series = series[series.index >= pd.Timestamp(ch["date_from"])]
    if ch.get("date_to"):
        series = series[series.index <= pd.Timestamp(ch["date_to"])]
    rule = {"day": "D", "week": "W-MON", "month": "MS"}[ch.get("granularity") or "day"]
    first_day, last_day = (series.index.min(), series.index.max()) if len(series) else (None, None)
    series = series.resample(rule, label="left", closed="left").sum()

    k1, k2, k3 = st.columns(3)
    k1.metric("Revenue in range", f"${series.sum():,.2f}", border=True)
    k2.metric("Days in range", f"{(last_day - first_day).days + 1:,}" if first_day is not None else "0", border=True)
    k3.metric("Granularity", (ch.get("granularity") or "day").capitalize(), border=True)
    if first_day is not None:
        st.caption(f"{first_day:%Y-%m-%d} → {last_day:%Y-%m-%d}")

    fig = px.line(series.rename("Revenue").reset_index(), x="date", y="Revenue",
                  labels={"date": "", "Revenue": "USD"}, markers=ch.get("granularity") != "day")
    fig.update_traces(line_width=1.5)
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch", key="revenue_chart")
    if ch.get("date_from") or ch.get("date_to") or ch.get("granularity") != "day":
        if st.button("Reset chart"):
            dash["chart"] = default_state()["chart"]
            st.rerun()

for name in TABLES:
    with tab[name.capitalize()]:
        shown = table_view(dash, name)
        total = len(_table(store, name))
        active = dash["filters"][name]
        srt = dash["sort"][name]

        a, b = st.columns([4, 1])
        a.metric(f"{name.capitalize()} shown", f"{len(shown):,} of {total:,}", border=True)
        if active or srt:
            if b.button("Clear filters", key=f"clear_{name}", width="stretch"):
                dash["filters"][name] = []
                dash["sort"][name] = None
                st.rerun()
        for f in active:
            neg = "NOT " if f.get("negate") else ""
            val = f.get("value")
            if isinstance(val, list) and len(val) > 8:
                val = f"{val[:8]} … ({len(val)} values)"
            st.caption(f"🔎 {neg}`{f['column']}` {f['op']} `{val if val is not None else ''}`")
        if srt:
            st.caption(f"↕ sorted by `{srt['column']}` {'ascending' if srt['ascending'] else 'descending'}")

        st.dataframe(shown, hide_index=True, width="stretch", height=min(560, 35 * (len(shown) + 1) + 3))
