"""Tools the AI can call.

The AI never sees raw tables and never counts by itself: it chooses a tool and arguments,
the pandas code below computes the exact answer, and the AI only phrases the result.

* Data tools (part 2) return facts.
* Control tools (part 3) change the dashboard state (store, view, filters, sorting, chart).
"""
import ast
import json
import operator
from functools import lru_cache

import pandas as pd

from analysis import STORES, prepare

TABLES = ["orders", "users", "books"]
VIEWS = ["Overview", "Revenue", "Books", "Users", "Orders"]
MAX_ROWS = 50


# ---------------------------------------------------------------------------
# Analysis-friendly tables (one row per order / user record / book)
# ---------------------------------------------------------------------------
def _authors_text(author_set):
    return ", ".join(sorted(author_set))


@lru_cache(maxsize=None)
def get_tables(store):
    us, orr, bo = prepare(store)

    books = pd.DataFrame({
        "book_id": bo["id"],
        "title": bo["title"],
        "authors": bo["author_set"].map(_authors_text),
        "n_authors": bo["author_set"].map(len),
        "genre": bo["genre"],
        "publisher": bo["publisher"],
        "year": pd.to_numeric(bo["year"], errors="coerce").astype("Int64"),   # a few years are blank
    })

    users = pd.DataFrame({
        "user_id": us["id"],
        "person_id": us["person"],
        "name": us["name"],
        "email": us["email"],
        "phone": us["phone"],
        "address": us["address"],
    })

    book_info = books.set_index("book_id")
    orders = pd.DataFrame({
        "order_id": orr["id"],
        "date": orr["date"].dt.strftime("%Y-%m-%d"),
        "year": orr["ts"].dt.year,
        "month": orr["ts"].dt.strftime("%Y-%m"),
        "weekday": orr["ts"].dt.day_name(),
        "hour": orr["ts"].dt.hour,
        "user_id": orr["user_id"],
        "person_id": orr["person"],
        "book_id": orr["book_id"],
        "title": orr["book_id"].map(book_info["title"]),
        "authors": orr["book_id"].map(book_info["authors"]),
        "genre": orr["book_id"].map(book_info["genre"]),
        "quantity": orr["quantity"],
        "unit_price_usd": orr["price_in_usd"].round(2),
        "paid_price": orr["paid_price"],
        "original_currency": orr["unit_price"].str.contains("€|EUR", case=False).map({True: "EUR", False: "USD"}),
        "shipping_address": orr["shipping"],
    })
    return {"orders": orders, "users": users, "books": books}


COLUMNS = {
    "orders": "order_id, date (YYYY-MM-DD), year, month (YYYY-MM), weekday (Monday..Sunday), hour (0-23), "
              "user_id, person_id, book_id, title, authors (comma-separated, sorted), genre, quantity, "
              "unit_price_usd, paid_price (USD = quantity*unit_price_usd), original_currency (USD/EUR), "
              "shipping_address (often empty)",
    "users": "user_id, person_id (same value for all aliases of one real person), name, email (lowercase), "
             "phone (10 digits), address (lowercase, may be empty)",
    "books": "book_id, title, authors (comma-separated, sorted), n_authors, genre, publisher, year (publication year)",
}


# ---------------------------------------------------------------------------
# Structured filters, shared by data tools and by the dashboard
# ---------------------------------------------------------------------------
FILTER_OPS = ["eq", "ne", "gt", "gte", "lt", "lte", "contains", "startswith", "endswith",
              "regex", "in", "between", "isnull", "notnull"]


def _as_list(v):
    return v if isinstance(v, list) else [v]


def _mask(df, f):
    col, op = f.get("column"), f.get("op")
    value = f.get("values") if f.get("values") not in (None, []) else f.get("value")
    if col not in df.columns:
        raise ValueError(f"unknown column '{col}'. Available: {', '.join(df.columns)}")
    if op not in FILTER_OPS:
        raise ValueError(f"unknown op '{op}'. Use one of: {', '.join(FILTER_OPS)}")

    s = df[col]
    is_text = s.dtype == object or pd.api.types.is_string_dtype(s)
    text = s.astype("string").str.lower() if is_text else None

    def cast(v):
        if is_text:
            return str(v).lower()
        return pd.to_numeric(v)

    if op == "isnull":
        m = s.isna()
    elif op == "notnull":
        m = s.notna()
    elif op in ("contains", "startswith", "endswith"):
        vals = [str(v).lower() for v in _as_list(value)]          # a list means "any of"
        src = text if is_text else s.astype("string")
        m = pd.Series(False, index=df.index)
        for v in vals:
            if op == "contains":
                m |= src.str.contains(v, regex=False).fillna(False)
            elif op == "startswith":
                m |= src.str.startswith(v).fillna(False)
            else:
                m |= src.str.endswith(v).fillna(False)
    elif op == "regex":
        m = s.astype("string").str.contains(str(value), case=False, regex=True).fillna(False)
    elif op == "in":
        vals = [cast(v) for v in _as_list(value)]
        m = (text if is_text else s).isin(vals)
    elif op == "between":
        lo, hi = _as_list(value)[:2]
        src = text if is_text else s
        m = (src >= cast(lo)) & (src <= cast(hi))
    else:
        src = text if is_text else s
        v = cast(value)
        m = {"eq": src == v, "ne": src != v, "gt": src > v,
             "gte": src >= v, "lt": src < v, "lte": src <= v}[op]
    m = m.fillna(False).astype(bool)
    return ~m if f.get("negate") else m


def apply_filters(df, filters):
    m = pd.Series(True, index=df.index)
    for f in filters or []:
        m &= _mask(df, f)
    return df[m]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _store(store):
    s = str(store).strip().upper().replace(" ", "")
    if s.isdigit():
        s = "STORE" + s
    if s not in STORES and s != "ALL":
        raise ValueError(f"unknown store '{store}'. Use one of: {', '.join(STORES)} or ALL")
    return s


def _table(store, table):
    if table not in TABLES:
        raise ValueError(f"unknown table '{table}'. Use one of: {', '.join(TABLES)}")
    store = _store(store)
    if store == "ALL":
        parts = []
        for s in STORES:
            t = get_tables(s)[table].copy()
            t.insert(0, "store", s)
            parts.append(t)
        return pd.concat(parts, ignore_index=True)
    return get_tables(store)[table]


def _records(df, limit=MAX_ROWS):
    out = df.head(limit).copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(2)
    return json.loads(out.to_json(orient="records", force_ascii=False))


def _person_block(store, person_id):
    t = get_tables(store)
    recs = t["users"][t["users"]["person_id"] == person_id]
    orders = t["orders"][t["orders"]["person_id"] == person_id]
    return {
        "person_id": int(person_id),
        "all_user_ids": sorted(recs["user_id"].astype(int).tolist()),
        "records": _records(recs.drop(columns="person_id")),
        "orders": int(len(orders)),
        "books_bought": int(orders["quantity"].sum()),
        "total_spent_usd": round(float(orders["paid_price"].sum()), 2),
        "first_order": orders["date"].min() if len(orders) else None,
        "last_order": orders["date"].max() if len(orders) else None,
    }


# ---------------------------------------------------------------------------
# Data tools (part 2)
# ---------------------------------------------------------------------------
def store_summary(store):
    """The dashboard answers for one store."""
    from analysis import solve
    store = _store(store)
    if store == "ALL":
        return {s: store_summary(s) for s in STORES}
    r = solve(store)
    t = get_tables(store)
    return {
        "store": store,
        "top5_days_by_revenue": [{"date": d, "revenue_usd": round(v, 2)} for d, v in zip(r["top5_days"], r["top5_revenue"])],
        "unique_real_users": int(r["unique_users"]),
        "user_records_including_aliases": int(len(t["users"])),
        "unique_author_sets": int(r["author_sets"]),
        "most_popular_author_set": r["top_authors"],
        "most_popular_author_set_books_sold": r["top_authors_sold"],
        "best_buyer_user_ids": r["best_buyer_ids"],
        "best_buyer_total_spent_usd": r["best_buyer_spend"],
        "total_revenue_usd": r["total_revenue"],
        "orders": int(r["orders"]),
        "books_sold": int(t["orders"]["quantity"].sum()),
        "first_order_date": t["orders"]["date"].min(),
        "last_order_date": t["orders"]["date"].max(),
    }


def query(store, table, filters=None, group_by=None, metric="count", metric_column=None,
          sort="desc", limit=10, columns=None):
    """Generic exact query: filter, then count/aggregate (optionally per group) or list rows."""
    df = apply_filters(_table(store, table), filters)
    limit = max(1, min(int(limit or 10), MAX_ROWS))
    metric = metric or "count"
    if metric not in ("count", "sum", "mean", "median", "min", "max", "nunique", "rows"):
        raise ValueError("metric must be one of count, sum, mean, median, min, max, nunique, rows")
    if metric not in ("count", "rows") and not metric_column:
        raise ValueError(f"metric '{metric}' needs metric_column")
    if metric_column and metric_column not in df.columns:
        raise ValueError(f"unknown metric_column '{metric_column}'. Available: {', '.join(df.columns)}")

    result = {"store": _store(store), "table": table, "matching_rows": int(len(df))}

    if metric == "rows":
        cols = [c for c in (columns or df.columns) if c in df.columns]
        rows = df[cols]
        if metric_column:
            rows = rows.sort_values(metric_column, ascending=(sort == "asc"))
        result["rows"] = _records(rows, limit)
        result["rows_shown"] = len(result["rows"])
        return result

    group_by = _as_list(group_by) if group_by else []
    for g in group_by:
        if g not in df.columns:
            raise ValueError(f"unknown group_by column '{g}'. Available: {', '.join(df.columns)}")

    if not group_by:
        if metric == "count":
            value = len(df)
        else:
            value = getattr(df[metric_column], metric)()
        if pd.isna(value):
            value = None
        elif metric in ("count", "nunique"):
            value = int(value)
        else:
            value = round(float(value), 2)
        result["value"] = value
        result["metric"] = metric if metric == "count" else f"{metric}({metric_column})"
        return result

    grouped = df.groupby(group_by, dropna=False)
    if metric == "count":
        agg = grouped.size().rename("value")
    else:
        agg = grouped[metric_column].agg(metric).rename("value")
    agg = agg.reset_index()
    agg = agg.sort_values(["value"] + group_by, ascending=[sort == "asc"] + [True] * len(group_by))
    result["metric"] = metric if metric == "count" else f"{metric}({metric_column})"
    result["groups_total"] = int(len(agg))
    result["groups"] = _records(agg, limit)
    if len(agg) and sort == "desc":
        top = agg["value"].iloc[0]
        result["ties_for_first"] = int((agg["value"] == top).sum())
    return result


def find_person(store, user_id=None, name=None, email=None, phone=None):
    """Look up a user (any alias) and return the real person with all ids, spending and orders."""
    store = _store(store)
    users = get_tables(store)["users"]
    m = pd.Series(True, index=users.index)
    if user_id is not None:
        m &= users["user_id"] == int(user_id)
    if name:
        m &= users["name"].str.lower().str.contains(str(name).lower(), regex=False)
    if email:
        m &= users["email"].str.contains(str(email).lower(), regex=False)
    if phone:
        digits = "".join(ch for ch in str(phone) if ch.isdigit())
        m &= users["phone"].str.contains(digits, regex=False)
    hits = users[m]
    persons = hits["person_id"].unique()
    out = {"store": store, "matching_records": int(len(hits)), "persons_found": int(len(persons))}
    if len(persons) == 0:
        return out
    out["persons"] = [_person_block(store, p) for p in persons[:10]]
    if len(persons) > 10:
        out["note"] = "only the first 10 persons are shown; refine the search"
    return out


def top_customers(store, n=5):
    """Real persons (aliases merged) ranked by total spending."""
    store = _store(store)
    orders = get_tables(store)["orders"]
    spend = orders.groupby("person_id")["paid_price"].sum().sort_values(ascending=False)
    n = max(1, min(int(n), 20))
    out = []
    for rank, (p, v) in enumerate(spend.head(n).items(), 1):
        b = _person_block(store, p)
        out.append({"rank": rank, "user_ids": b["all_user_ids"], "name": b["records"][0]["name"],
                    "total_spent_usd": round(float(v), 2), "orders": b["orders"]})
    return {"store": store, "customers": out}


def author_stats(store, author):
    """Books, copies sold and revenue of every author whose name contains `author`."""
    store = _store(store)
    t = get_tables(store)
    books, orders = t["books"], t["orders"]
    names = sorted({a.strip() for s in books["authors"] for a in s.split(",")})
    hits = [a for a in names if str(author).lower() in a.lower()]
    out = {"store": store, "authors_found": len(hits), "authors": []}
    for a in hits[:10]:
        bk = books[books["authors"].map(lambda s: a in [x.strip() for x in s.split(",")])]
        od = orders[orders["book_id"].isin(bk["book_id"])]
        out["authors"].append({
            "author": a,
            "books_written": int(len(bk)),
            "solo_books": int((bk["n_authors"] == 1).sum()),
            "coauthored_books": int((bk["n_authors"] > 1).sum()),
            "copies_sold": int(od["quantity"].sum()),
            "revenue_usd": round(float(od["paid_price"].sum()), 2),
            "titles": bk["title"].tolist()[:20],
        })
    return out


def list_values(store, table, column, filters=None, limit=1000):
    """Distinct values of one column (for decisions that need world knowledge, e.g. names in titles)."""
    df = apply_filters(_table(store, table), filters)
    if column not in df.columns:
        raise ValueError(f"unknown column '{column}'. Available: {', '.join(df.columns)}")
    vals = df[column].dropna().drop_duplicates().sort_values()
    limit = max(1, min(int(limit), 1000))
    return {"distinct_values": int(len(vals)), "values": vals.head(limit).tolist()}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.Mod: operator.mod}


def calculate(expression):
    """Safe arithmetic on numbers returned by other tools (the model must not do math itself)."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("round", "abs") and node.args:
            return {"round": round, "abs": abs}[node.func.id](*[ev(a) for a in node.args])
        raise ValueError("only numbers, + - * / ** %, round() and abs() are allowed")
    value = ev(ast.parse(str(expression), mode="eval"))
    return {"expression": expression, "result": round(value, 6)}


# ---------------------------------------------------------------------------
# Control tools (part 3): they change `state`, the dashboard redraws from it
# ---------------------------------------------------------------------------
def default_state():
    return {
        "store": "STORE1",
        "view": "Overview",
        "filters": {t: [] for t in TABLES},
        "sort": {t: None for t in TABLES},
        "chart": {"date_from": None, "date_to": None, "granularity": "day"},
    }


def set_store(state, store):
    store = _store(store)
    if store == "ALL":
        raise ValueError("the dashboard shows one store at a time; pick STORE1..STORE5")
    state["store"] = store
    return {"ok": True, "store": store}


def set_view(state, view):
    match = [v for v in VIEWS if v.lower() == str(view).lower()]
    if not match:
        raise ValueError(f"unknown view '{view}'. Use one of: {', '.join(VIEWS)}")
    state["view"] = match[0]
    return {"ok": True, "view": match[0]}


def filter_table(state, table, filters, mode="replace"):
    df = _table(state["store"], table)
    new = filters if mode == "replace" else state["filters"][table] + filters
    shown = apply_filters(df, new)          # validates columns/ops before saving
    state["filters"][table] = new
    state["view"] = table.capitalize()
    return {"ok": True, "store": state["store"], "table": table, "active_filters": new,
            "rows_shown": int(len(shown)), "rows_total": int(len(df))}


def clear_filters(state, table="all"):
    for t in (TABLES if table == "all" else [table]):
        state["filters"][t] = []
        state["sort"][t] = None
    return {"ok": True, "cleared": table}


def sort_table(state, table, column, ascending=True):
    df = _table(state["store"], table)
    if column not in df.columns:
        raise ValueError(f"unknown column '{column}'. Available: {', '.join(df.columns)}")
    state["sort"][table] = {"column": column, "ascending": bool(ascending)}
    state["view"] = table.capitalize()
    return {"ok": True, "table": table, "sorted_by": column, "ascending": bool(ascending)}


def set_chart(state, date_from=None, date_to=None, granularity=None):
    if granularity and granularity not in ("day", "week", "month"):
        raise ValueError("granularity must be day, week or month")
    for d in (date_from, date_to):
        if d:
            pd.Timestamp(d)                   # validates the date
    state["chart"] = {"date_from": date_from, "date_to": date_to,
                      "granularity": granularity or state["chart"].get("granularity", "day")}
    state["view"] = "Revenue"
    return {"ok": True, "chart": state["chart"]}


def reset_dashboard(state, confirm=True):
    store = state["store"]
    state.clear()
    state.update(default_state())
    state["store"] = store
    return {"ok": True}


def table_view(state, table):
    """What the dashboard shows for a table: current store, filters and sort applied."""
    df = apply_filters(_table(state["store"], table), state["filters"][table])
    srt = state["sort"][table]
    if srt:
        df = df.sort_values(srt["column"], ascending=srt["ascending"])
    return df


# ---------------------------------------------------------------------------
# Tool descriptions for the model (JSON schema)
# ---------------------------------------------------------------------------
_STORE = {"type": "string", "description": f"One of {', '.join(STORES)} (or ALL for every store together where allowed)."}
_TABLE = {"type": "string", "enum": TABLES}
_FILTERS = {
    "type": "array",
    "description": "Conditions combined with AND. String comparisons are case-insensitive. "
                   "Use `value` for one value, `values` for several (in, between, or any-of for "
                   "contains/startswith/endswith). Set negate=true to exclude matching rows.",
    "items": {
        "type": "object",
        "properties": {
            "column": {"type": "string"},
            "op": {"type": "string", "enum": FILTER_OPS},
            "value": {"type": "string", "description": "Single value (numbers as text are fine, e.g. '2020')."},
            "values": {"type": "array", "items": {"type": "string"},
                       "description": "Several values: for in, between [low, high], or any-of for contains/startswith/endswith."},
            "negate": {"type": "boolean"},
        },
        "required": ["column", "op"],
    },
}

DATA_TOOLS = {
    "store_summary": {
        "fn": store_summary,
        "description": "Headline numbers of a store: top 5 revenue days, unique real users, unique author sets, "
                       "most popular author set, best buyer ids, total revenue, orders, date range. Use ALL for every store.",
        "params": {"store": _STORE},
        "required": ["store"],
    },
    "query": {
        "fn": query,
        "description": "Exact filtered count/aggregation/listing over one table. Tables and columns: "
                       + " | ".join(f"{k}: {v}" for k, v in COLUMNS.items())
                       + ". Revenue = sum(paid_price) on orders. Real users = nunique(person_id). "
                         "Use metric='rows' to list rows. store=ALL adds a 'store' column.",
        "params": {
            "store": _STORE, "table": _TABLE, "filters": _FILTERS,
            "group_by": {"type": "array", "items": {"type": "string"}, "description": "Columns to group by."},
            "metric": {"type": "string", "enum": ["count", "sum", "mean", "median", "min", "max", "nunique", "rows"]},
            "metric_column": {"type": "string", "description": "Column for sum/mean/median/min/max/nunique, or sort column for rows."},
            "sort": {"type": "string", "enum": ["desc", "asc"]},
            "limit": {"type": "integer", "description": "Max groups/rows returned (1-50)."},
            "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to return when metric='rows'."},
        },
        "required": ["store", "table"],
    },
    "find_person": {
        "fn": find_person,
        "description": "Find users by user_id, part of name, email or phone. Returns each real person with ALL alias user_ids, "
                       "their records, orders count, books bought and total spending.",
        "params": {"store": _STORE, "user_id": {"type": "integer"}, "name": {"type": "string"},
                   "email": {"type": "string"}, "phone": {"type": "string"}},
        "required": ["store"],
    },
    "top_customers": {
        "fn": top_customers,
        "description": "Real persons (aliases merged) ranked by total spending, with all their user_ids.",
        "params": {"store": _STORE, "n": {"type": "integer"}},
        "required": ["store"],
    },
    "author_stats": {
        "fn": author_stats,
        "description": "For authors whose name contains the text: books written (solo/co-authored), copies sold, revenue, titles.",
        "params": {"store": _STORE, "author": {"type": "string"}},
        "required": ["store", "author"],
    },
    "list_values": {
        "fn": list_values,
        "description": "Distinct values of a column (optionally filtered). Use when a decision needs world knowledge, "
                       "e.g. which book titles contain a person's name; then filter with op 'in' on the exact values.",
        "params": {"store": _STORE, "table": _TABLE, "column": {"type": "string"}, "filters": _FILTERS,
                   "limit": {"type": "integer"}},
        "required": ["store", "table", "column"],
    },
    "calculate": {
        "fn": calculate,
        "description": "Arithmetic on numbers from other tools (differences, ratios, percentages). Never do math in your head.",
        "params": {"expression": {"type": "string", "description": "e.g. '(739353.52-668103.35)/668103.35*100'"}},
        "required": ["expression"],
    },
}

CONTROL_TOOLS = {
    "set_store": {
        "fn": set_store,
        "description": "Switch the dashboard to another store.",
        "params": {"store": {"type": "string", "enum": STORES}},
        "required": ["store"],
    },
    "set_view": {
        "fn": set_view,
        "description": f"Open a dashboard view: {', '.join(VIEWS)}.",
        "params": {"view": {"type": "string", "enum": VIEWS}},
        "required": ["view"],
    },
    "filter_table": {
        "fn": filter_table,
        "description": "Filter the books, users or orders table shown on the dashboard (current store) and open it. "
                       "mode='replace' sets these filters, mode='add' adds them to the active ones. "
                       "To HIDE matching rows use negate=true. Columns: "
                       + " | ".join(f"{k}: {v}" for k, v in COLUMNS.items()),
        "params": {"table": _TABLE, "filters": _FILTERS, "mode": {"type": "string", "enum": ["replace", "add"]}},
        "required": ["table", "filters"],
    },
    "clear_filters": {
        "fn": clear_filters,
        "description": "Remove filters and sorting from one table or from all tables.",
        "params": {"table": {"type": "string", "enum": TABLES + ["all"]}},
        "required": [],
    },
    "sort_table": {
        "fn": sort_table,
        "description": "Sort a dashboard table by a column.",
        "params": {"table": _TABLE, "column": {"type": "string"}, "ascending": {"type": "boolean"}},
        "required": ["table", "column"],
    },
    "set_chart": {
        "fn": set_chart,
        "description": "Change the revenue chart: date range (YYYY-MM-DD, null = no limit) and granularity day/week/month. Opens the Revenue view.",
        "params": {"date_from": {"type": "string", "description": "YYYY-MM-DD; omit for no lower limit"},
                   "date_to": {"type": "string", "description": "YYYY-MM-DD; omit for no upper limit"},
                   "granularity": {"type": "string", "enum": ["day", "week", "month"]}},
        "required": [],
    },
    "reset_dashboard": {
        "fn": reset_dashboard,
        "description": "Reset view, filters, sorting and chart settings (keeps the store).",
        "params": {"confirm": {"type": "boolean", "description": "Always true."}},
        "required": [],
    },
}


def tool_specs():
    """Provider-neutral tool list: name, description, JSON schema."""
    specs = []
    for name, t in {**DATA_TOOLS, **CONTROL_TOOLS}.items():
        specs.append({
            "name": name,
            "description": t["description"],
            "schema": {"type": "object", "properties": t["params"], "required": t["required"]},
        })
    return specs


def run_tool(name, args, state):
    """Execute one tool call. Errors are returned to the model so it can correct itself."""
    try:
        if name in DATA_TOOLS:
            result = DATA_TOOLS[name]["fn"](**args)
        elif name in CONTROL_TOOLS:
            result = CONTROL_TOOLS[name]["fn"](state, **args)
        else:
            result = {"error": f"unknown tool '{name}'"}
    except Exception as e:  # noqa: BLE001 - every error goes back to the model as text
        result = {"error": f"{type(e).__name__}: {e}"}
    return json.loads(json.dumps(result, default=str))
