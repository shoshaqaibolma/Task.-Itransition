import re
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from dateutil import parser
stores = ["STORE1", "STORE2", "STORE3", "STORE4", r"STORE5"]

def load(store):

     us = pd.read_csv(store + "/users.csv", dtype=str, keep_default_na=False)

     us["id"] = us["id"].astype(int)

     orr = pd.read_parquet(store + "/orders.parquet")

     with open(store + "/books.yaml", encoding="utf-8") as f:

          bo = pd.DataFrame(yaml.safe_load(f))

     bo.columns = [c.replace(":", "") for c in bo.columns] 

     return us, orr, bo

  
  



def price_to_usd(prices):

     s = []

     for i in prices:

          m = re.search(r"(\d+)(?:[.¢€$]+(\d+))?", i)

          if m.group(2):

               num = float(m.group(1) + "." + m.group(2))

          else:

               num = float(m.group(1))

          if "€" in i or "EUR" in i.upper():

               num *= 1.2

          s.append(num)

     return s

  
  




SENTINEL = datetime(1900, 1, 1)

  
  

def parse_time(stamps):

     time_norm = []

     for i in stamps:

          clean = re.sub(r"\b([AaPp])\.?[Mm]\.?", lambda m: m.group(1).upper() + "M", i)

          clean = re.sub(r"[;,]", " ", clean).strip()

          dayfirst = re.search(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b", clean) is not None

          time_norm.append(parser.parse(clean, default=SENTINEL, dayfirst=dayfirst))

     return pd.to_datetime(time_norm)

  
  


def clean_orders(orr):

     orr = orr.drop_duplicates()

     orr = orr.drop_duplicates(subset="id")

     orr["shipping"] = orr["shipping"].str.strip().replace({"": pd.NA, "NULL": pd.NA})

     return orr
     



def daily_rev(orr):
     daily = orr.groupby("date")["paid_price"].sum().sort_index()
     daily = daily.asfreq("D", fill_value=0)     
     return daily.round(2)  



def clean_users(us):

     us["name"] = us["name"].str.strip().str.replace(r"\s+", " ", regex=True)

     us["address"] = (us["address"].str.strip().str.lower()

                         .str.replace(r"\s+", " ", regex=True)

                         .replace({"": pd.NA, "null": pd.NA}))

     us["phone"] = us["phone"].str.replace(r"\D", "", regex=True)

     us["email"] = us["email"].str.strip().str.lower()

     return us

  
  

def find_persons(us):

     fields = ["name", "address", "phone", "email"]

     us["person"] = us["id"]

     while True:

          before = us["person"].copy()

          for f in fields:

               others = [x for x in fields if x != f]

               us["person"] = us.groupby(others, dropna=False)["person"].transform("min")

          if us["person"].equals(before):

               break

     return us

  
  



def author_sets(bo):

     sets = []

     for i in bo["author"]:

          names = [x.strip() for x in i.split(",")]

          sets.append(frozenset(names))

     bo["author_set"] = sets

     return bo

  

def solve(store):

     us, orr, bo = load(store)

  

     orr["price_in_usd"] = price_to_usd(orr["unit_price"])

     orr["paid_price"] = (orr["quantity"] * orr["price_in_usd"]).round(2)

     orr["ts"] = parse_time(orr["timestamp"])

     orr["date"] = orr["ts"].dt.normalize()

     orr = clean_orders(orr)

  

     us = find_persons(clean_users(us))

     bo = author_sets(bo)

  

     orr["person"] = orr["user_id"].map(us.set_index("id")["person"])

     orr["author_set"] = orr["book_id"].map(bo.set_index("id")["author_set"])

  

     daily = daily_rev(orr)

     sold = orr.groupby("author_set")["quantity"].sum()

     spend = orr.groupby("person")["paid_price"].sum()

     top = spend.idxmax()

  

     return {
        "top5_days": [d.strftime("%Y-%m-%d") for d in daily.nlargest(5).index],
        "top5_revenue": daily.nlargest(5).tolist(),
        "unique_users": us["person"].nunique(),
        "author_sets": bo["author_set"].nunique(),
        "top_authors": sorted(sold.idxmax()),
        "top_authors_sold": int(sold.max()),
        "best_buyer_ids": sorted(us.loc[us["person"] == top, "id"].tolist()),
        "best_buyer_spend": round(spend.max(), 2),
        "total_revenue": round(orr["paid_price"].sum(), 2),
        "orders": len(orr),
        "daily": daily,
    }

    

  
  

results = {}

for store in stores:

     results[store] = solve(store)

     r = results[store]

     print(store)

     print("  top 5 days:   ", r["top5_days"])

     print("  unique users: ", r["unique_users"])

     print("  author sets:  ", r["author_sets"])

     print("  top authors:  ", r["top_authors"])

     print("  best buyer:   ", r["best_buyer_ids"])
     
for store in stores:

     plt.figure(figsize=(12, 3))

     plt.plot(results[store]["daily"].index, results[store]["daily"].values, linewidth=0.8)

     plt.title(store + " — daily revenue")

     plt.ylabel("USD")

     plt.grid(alpha=0.3)

     plt.show()