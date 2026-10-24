# Task 4 — Bookstore Analytics

Dashboard: **<add your Streamlit link here>**

Five bookstores (`STORE1` … `STORE5`, each with `users.csv`, `orders.parquet`, `books.yaml`) are cleaned and analysed,
shown in a Streamlit dashboard, and an AI chat answers questions about the data and controls the dashboard.

## Part 1 — analysis (`analysis.py`)
* **Prices** come in ~50 formats (`$12.50`, `12.50 USD`, `EUR 12.5`, `€12¢50`, `29€50¢`, `12.$`); `¢` is a decimal
  separator. Euro prices are converted at 1 EUR = 1.2 USD. `paid_price = quantity × unit price`.
* **Timestamps** mix ISO, `MM/DD/YY`, `DD.MM.YYYY`, `DD-Mon-YYYY`, ctime, 12/24 h clocks and `A.M./pm`.
  Dotted dates are day-first, slash dates month-first (checked on the data). A missing date would show up as 1900,
  never as "today", so there is no artificial spike at the end of the revenue chart.
* **Users**: two records are the same person when they agree on all of name/address/phone/email except one.
  Labels are propagated through groupings until stable, so chains (A~B, B~C) merge too.
* **Author sets**: `frozenset` of the comma-separated authors, so the order of names does not matter.

Run `python analysis.py` to print the answers for every store.

## Part 2 — questions in natural language (`tools.py`, `llm.py`)
The model never sees raw tables and never counts. It chooses a tool, pandas computes the exact answer, and the
model only phrases it. Every answer shows the tool calls it used (🔧 expander under the message).

| Tool | Purpose |
|---|---|
| `store_summary` | the dashboard answers for a store (or ALL) |
| `query` | exact filtered count / sum / mean / min / max / nunique / listing, optionally grouped |
| `find_person` | a user by id, name, e-mail or phone, with all alias ids, orders and spending |
| `top_customers` | real persons ranked by spending |
| `author_stats` | books, copies sold and revenue of an author |
| `list_values` | distinct values of a column (for judgment calls such as "names in titles") |
| `calculate` | safe arithmetic on tool results (no math in the model's head) |

The system prompt forbids numbers that did not come from a tool and tells the model to say so when the data cannot
answer.

## Part 3 — the AI controls the dashboard
Control tools change one state object that the dashboard is drawn from: `set_store`, `set_view`,
`filter_table` (books / users / orders, 14 operators, negation), `sort_table`, `clear_filters`, `set_chart`
(date range, day/week/month), `reset_dashboard`. Examples:
* "Select users with e-mails started with s or p" → `filter_table(users, email startswith [s, p])`
* "Filter out books that have people names in the title" → `list_values(books.title)`, the model picks the titles,
  then `filter_table(books, title in [...])`
* "Show the revenue chart by month for 2025" → `set_chart(2025-01-01, 2025-12-31, month)`

## AI provider
Default is **Google Gemini** (`gemini-3.8-flash`, free tier) through its OpenAI-compatible endpoint.
Claude (Anthropic) or any OpenAI-compatible API can be used instead — see `.streamlit/secrets.toml.example`.

## Run locally
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then put your free Gemini key in it
streamlit run app.py
```
