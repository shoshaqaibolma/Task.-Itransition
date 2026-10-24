import plotly.express as px
import streamlit as st

from analysis import solve

st.set_page_config(page_title="Bookstore Analytics", layout="wide")

STORES = ["STORE1", "STORE2", "STORE3", "STORE4", "STORE5"]


@st.cache_data
def get_results(store):
    return solve(store)


st.title(" Bookstore Analytics")
st.caption("Task 4 · daily revenue, real users, author sets and best buyers for each store")

tabs = st.tabs(STORES)

for tab, store in zip(tabs, STORES):
    with tab:
        r = get_results(store)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unique users", f"{r['unique_users']:,}")
        c2.metric("Unique author sets", f"{r['author_sets']:,}")
        c3.metric("Total revenue", f"${r['total_revenue']:,.2f}")
        c4.metric("Orders", f"{r['orders']:,}")

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Top 5 days by revenue")
            st.dataframe(
                {"Date": r["top5_days"], "Revenue, $": [f"{v:,.2f}" for v in r["top5_revenue"]]},
                hide_index=True,
                width="stretch",
            )
        with right:
            st.subheader("Most popular author(s)")
            st.markdown("**" + ", ".join(r["top_authors"]) + "**")
            st.caption(f"{r['top_authors_sold']} books sold")

            st.subheader("Best buyer")
            st.code(str(r["best_buyer_ids"]), language=None)
            st.caption(f"${r['best_buyer_spend']:,.2f} spent · ids of the same person (aliases)")

        st.subheader("Daily revenue")
        df = r["daily"].rename("Revenue").reset_index()
        fig = px.line(df, x="date", y="Revenue", labels={"date": "", "Revenue": "USD"})
        fig.update_traces(line_width=1)
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch", key="chart_" + store)

st.divider()
st.subheader("💬 Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about the data…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    answer = "Chat will be connected in parts 2 and 3."
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
