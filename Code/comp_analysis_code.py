"""
Baja SAE insights - single-file tkinter GUI with Plotly Express charts.

Run:  python baja_tool.py
pip install pandas numpy openpyxl plotly
Optional: statsmodels (trendlines); requests beautifulsoup4 lxml (scraping)

Input: .xlsx with one sheet per competition (headers may differ between comps).
"""
import re
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import plotly.express as px

try:  # optional: enables OLS trendlines on scatter plots
    import statsmodels.api  # noqa: F401
    _TREND = "ols"
except ImportError:
    _TREND = None


# --------------------------------------------------------------------------
# 1. Column normalisation
# --------------------------------------------------------------------------
# (canonical name, regex on lower-cased header with "(points)" stripped)
# Order matters: first match wins.
COLUMN_PATTERNS = [
    ("overall_dynamic", r"^overall dynamic"),
    ("overall_static", r"^overall static"),
    ("overall", r"^overall"),
    ("cost", r"^cost"),
    ("design", r"^design"),
    ("business", r"^business|^sales"),
    ("accel", r"^accel"),
    ("maneuver", r"^maneuver"),
    ("hill_climb", r"^hill"),
    ("sled_pull", r"pull|^sled|^traction"),
    ("rock_crawl", r"^rock"),
    ("susp_traction", r"^s&t|^suspension"),
    ("endurance", r"^endurance"),
    ("rank", r"^rank"),
    ("car", r"^car"),
    ("school", r"^school"),
    ("team", r"^team"),
    ("adjustments", r"^adjust"),
]

EVENTS = ["cost", "design", "business", "accel", "maneuver",
          "hill_climb", "sled_pull", "rock_crawl", "susp_traction", "endurance"]
DYNAMIC = {"accel", "maneuver", "hill_climb", "sled_pull", "rock_crawl", "susp_traction", "endurance"}

# Fallback max points if the header has no "(N)" suffix
DEFAULT_MAX = {"overall": 1000, "overall_dynamic": 300, "overall_static": 300,
               "cost": 100, "design": 150, "business": 50, "accel": 75,
               "maneuver": 75, "hill_climb": 75, "sled_pull": 75, "rock_crawl": 75,
               "susp_traction": 75, "endurance": 400}


def canonical(header: str):
    h = re.sub(r"\(.*?\)", "", str(header)).strip().lower()
    # strip trailing "event" etc. so "Cost Event" -> "cost event" still matches ^cost
    for name, pat in COLUMN_PATTERNS:
        if re.search(pat, h):
            return name
    return None


def max_points(header: str, name: str):
    m = re.search(r"\((\d+)\)", str(header))
    return float(m.group(1)) if m else DEFAULT_MAX.get(name, np.nan)


def clean_sheet(df: pd.DataFrame, comp: str):
    """Rename columns to canonical names; return (clean df, {event: max_points})."""
    rename, maxes = {}, {}
    for col in df.columns:
        name = canonical(col)
        if name and name not in rename.values():
            rename[col] = name
            maxes[name] = max_points(col, name)
    ignore = r"registered|passed tech|public comments"
    clean_sheet.unmatched = [str(c) for c in df.columns
                             if c not in rename and not re.search(ignore, str(c).lower())
                             and not str(c).lower().startswith("unnamed")]
    out = df[list(rename)].rename(columns=rename).copy()
    for c in out.columns:
        if c in maxes and c not in ("rank", "car"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out.insert(0, "competition", comp)
    out = out.dropna(subset=["overall"]) if "overall" in out else out
    return out, maxes


def load_workbook(path: str, treat_zero_as_missing: bool = True):
    """Return (long dataframe of all comps, {comp: {event: max_pts}})."""
    sheets = pd.read_excel(path, sheet_name=None)
    frames, all_max, unmatched = [], {}, {}
    for comp, raw in sheets.items():
        if raw.empty:
            continue
        df, maxes = clean_sheet(raw, str(comp))
        if treat_zero_as_missing:
            # A 0 in a single event almost always means DNS/DQ/not run, not a
            # genuine zero-skill result; it would wreck rank correlations.
            for ev in EVENTS:
                if ev in df:
                    df.loc[df[ev] == 0, ev] = np.nan
        frames.append(df)
        all_max[str(comp)] = maxes
        if clean_sheet.unmatched:
            unmatched[str(comp)] = clean_sheet.unmatched
    data = pd.concat(frames, ignore_index=True)
    data.attrs["unmatched"] = unmatched
    return data, all_max


# --------------------------------------------------------------------------
# 2. Normalisation so competitions are comparable
# --------------------------------------------------------------------------
def add_normalised(data: pd.DataFrame):
    """
    Adds, for every event + overall:
      <x>_pct  : percentile within that competition (1.0 = best). Rank-based, so
                 it is immune to different point scales / different course difficulty.
    """
    cols = [c for c in EVENTS + ["overall", "overall_dynamic", "overall_static"]
            if c in data.columns]
    pct = data.groupby("competition")[cols].rank(pct=True, na_option="keep")
    pct.columns = [f"{c}_pct" for c in cols]
    return pd.concat([data, pct], axis=1)


def pct_frame(data: pd.DataFrame, cols):
    return data[[f"{c}_pct" for c in cols if f"{c}_pct" in data.columns]].rename(
        columns=lambda c: c[:-4])


# --------------------------------------------------------------------------
# 3. Analyses
# --------------------------------------------------------------------------
def pooled_corr(data, cols, min_n=15):
    """Spearman correlation on within-competition percentiles, pooled across comps."""
    p = pct_frame(data, cols)
    corr = p.corr(method="spearman", min_periods=min_n)
    n = p.notna().astype(int).T.dot(p.notna().astype(int))
    return corr, n


def pairwise_table(data, cols, min_n=15):
    """Every event pair: pooled rho, n, and the spread of rho across competitions."""
    corr, n = pooled_corr(data, cols, min_n)
    rows = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            per_comp = []
            for _, g in data.groupby("competition"):
                if a in g and b in g and g[[a, b]].dropna().shape[0] >= 10:
                    per_comp.append(g[[a, b]].corr(method="spearman").iloc[0, 1])
            rows.append({
                "event_a": a, "event_b": b,
                "rho_pooled": corr.loc[a, b], "n": int(n.loc[a, b]),
                "n_comps": len(per_comp),
                "rho_min_comp": np.min(per_comp) if per_comp else np.nan,
                "rho_max_comp": np.max(per_comp) if per_comp else np.nan,
            })
    return pd.DataFrame(rows).sort_values("rho_pooled", ascending=False)


def overall_drivers(data, min_n=15):
    """
    How strongly does each event predict overall finish?
    'rho_overall'      : raw correlation with overall (inflated, event is part of overall)
    'rho_ex_event'     : correlation with overall points MINUS that event's points
                         (fairer: removes the part-whole effect)
    """
    rows = []
    for ev in EVENTS:
        if ev not in data or data[ev].notna().sum() < min_n:
            continue
        sub = data[["competition", "overall", ev]].dropna()
        ex = sub["overall"] - sub[ev]
        tmp = sub.assign(ex=ex)
        pct = tmp.groupby("competition")[[ev, "overall", "ex"]].rank(pct=True)
        rows.append({
            "event": ev, "n": len(sub),
            "rho_overall": pct[ev].corr(pct["overall"], method="spearman"),
            "rho_ex_event": pct[ev].corr(pct["ex"], method="spearman"),
        })
    return pd.DataFrame(rows).sort_values("rho_ex_event", ascending=False)


def top_vs_field(data, top_k=10):
    """Mean percentile in each event for top-k overall finishers vs the rest."""
    d = data.copy()
    d["is_top"] = d.groupby("competition")["overall"].rank(ascending=False) <= top_k
    cols = [f"{e}_pct" for e in EVENTS if f"{e}_pct" in d]
    t = d.groupby("is_top")[cols].mean().T
    t.columns = ["rest", f"top{top_k}"]
    t.index = [i[:-4] for i in t.index]
    t["gap"] = t[f"top{top_k}"] - t["rest"]
    return t.sort_values("gap", ascending=False)


def points_leverage(data, all_max):
    """
    Where are points actually won/lost? For each event:
      max_pts       : available points
      mean_pts_lost : average points teams leave on the table
      std_pts       : standard deviation of points (how much the event separates teams)
      top10_vs_rest : mean points advantage of top-10 over everyone else
    """
    d = data.copy()
    d["is_top"] = d.groupby("competition")["overall"].rank(ascending=False) <= 10
    rows = []
    for ev in EVENTS:
        if ev not in d or d[ev].notna().sum() < 10:
            continue
        mx = np.nanmedian([m[ev] for m in all_max.values() if ev in m])
        s = d[ev]
        rows.append({
            "event": ev, "max_pts": mx,
            "mean_pts": s.mean(), "mean_pts_lost": mx - s.mean(),
            "std_pts": s.std(),
            "top10_vs_rest": s[d.is_top].mean() - s[~d.is_top].mean(),
        })
    return pd.DataFrame(rows).sort_values("top10_vs_rest", ascending=False)


def standardized_regression(data):
    """
    Standardised OLS of overall percentile on event percentiles (rows with all
    events present). Betas show each event's *unique* contribution after
    accounting for the others (e.g. hill climb vs accel overlap).
    """
    evs = [e for e in EVENTS if f"{e}_pct" in data and data[f"{e}_pct"].notna().mean() > 0.5]
    sub = data[["overall_pct"] + [f"{e}_pct" for e in evs]].dropna()
    if len(sub) < 3 * len(evs):
        return None
    X = (sub[[f"{e}_pct" for e in evs]] - sub[[f"{e}_pct" for e in evs]].mean()) / \
        sub[[f"{e}_pct" for e in evs]].std()
    y = (sub["overall_pct"] - sub["overall_pct"].mean()) / sub["overall_pct"].std()
    beta, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
    r2 = 1 - ((y - X.values @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    out = pd.Series(beta, index=evs, name="std_beta").sort_values(ascending=False).to_frame()
    out.attrs["r2"], out.attrs["n"] = r2, len(sub)
    return out


# --------------------------------------------------------------------------
# 4. Scraper  (UNTESTED against the live site - see notes)
# --------------------------------------------------------------------------
def scrape_to_excel(urls, out_path, sheet_names=None):
    """
    Pull the results table from each URL into one workbook (one sheet per comp).
    Strategy: use any <table> on the page via pandas.read_html and keep the
    largest one that has a 'Rank' column. If the page links directly to an
    .xlsx/.csv export, that file is used instead.

    NOTE: the Baja SAE results site may render tables with JavaScript, in which
    case read_html returns nothing - then download the per-competition Excel/CSV
    export from the site manually, or point Playwright/Selenium at the page.
    """
    import io
    import requests
    from bs4 import BeautifulSoup

    headers = {"User-Agent": "Mozilla/5.0 (baja-analysis; team research)"}
    with pd.ExcelWriter(out_path) as xw:
        for i, url in enumerate(urls):
            name = (sheet_names[i] if sheet_names else None) or f"comp_{i + 1}"
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            df = None
            # direct file link?
            if url.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(r.content))
            elif url.lower().endswith(".csv"):
                df = pd.read_csv(io.BytesIO(r.content))
            else:
                soup = BeautifulSoup(r.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    if re.search(r"\.(xlsx|xls|csv)(\?|$)", a["href"], re.I):
                        from urllib.parse import urljoin
                        fr = requests.get(urljoin(url, a["href"]), headers=headers, timeout=30)
                        fr.raise_for_status()
                        df = (pd.read_csv if a["href"].lower().split("?")[0].endswith("csv")
                              else pd.read_excel)(io.BytesIO(fr.content))
                        break
                if df is None:
                    tables = [t for t in pd.read_html(io.StringIO(r.text))
                              if any("rank" in str(c).lower() for c in t.columns)]
                    if tables:
                        df = max(tables, key=len)
            if df is None:
                print(f"[warn] no results table found at {url}", file=sys.stderr)
                continue
            df.to_excel(xw, sheet_name=str(name)[:31], index=False)
            print(f"[ok] {name}: {len(df)} teams from {url}")



# --------------------------------------------------------------------------
# 5. Plotly Express figures
# --------------------------------------------------------------------------

def available_events(data, min_n=15):
    return [e for e in EVENTS if e in data and data[e].notna().sum() >= min_n]


def heatmap_fig(data):
    cols = available_events(data) + ["overall"]
    corr, _ = pooled_corr(data, cols)
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Event correlations (Spearman, within-competition percentiles)")
    fig.update_layout(coloraxis_colorbar_title="rho")
    return fig


def scatter_fig(data, x, y, mode="Percentile"):
    """mode: 'Percentile' (comparable across comps) or 'Points' (raw scores)."""
    xc, yc = (f"{x}_pct", f"{y}_pct") if mode == "Percentile" else (x, y)
    hover = [c for c in ("team", "school", "overall") if c in data and c not in (xc, yc)]
    sub = data[[xc, yc, "competition"] + hover].dropna(subset=[xc, yc])
    if sub.empty:
        raise ValueError(f"No teams have both {x} and {y}.")
    rho = sub[xc].corr(sub[yc], method="spearman")
    fig = px.scatter(sub, x=xc, y=yc, color="competition", hover_data=hover,
                     opacity=0.75,
                     trendline=_TREND, trendline_scope="overall",
                     trendline_color_override="black",
                     title=f"{y} vs {x}  (Spearman rho = {rho:.2f}, n = {len(sub)})",
                     labels={xc: f"{x} ({mode.lower()})", yc: f"{y} ({mode.lower()})"})
    return fig


def drivers_fig(drivers):
    long = drivers.melt(id_vars="event", value_vars=["rho_overall", "rho_ex_event"],
                        var_name="measure", value_name="rho")
    return px.bar(long, x="event", y="rho", color="measure", barmode="group",
                  title="Correlation of each event with overall finish "
                        "(ex_event removes the event's own points from overall)")


def gap_fig(gap, top_k):
    long = gap.reset_index(names="event").melt(
        id_vars="event", value_vars=["rest", f"top{top_k}"],
        var_name="group", value_name="mean_percentile")
    fig = px.bar(long, x="event", y="mean_percentile", color="group", barmode="group",
                 title=f"Mean event percentile: top-{top_k} overall finishers vs the rest")
    fig.update_yaxes(range=[0, 1])
    return fig


def leverage_fig(lev):
    return px.scatter(
        lev, x="mean_pts_lost", y="top10_vs_rest", size="std_pts", text="event",
        color="event", size_max=45,
        title="Where to focus: points teams leave on the table vs how much top teams "
              "out-score the field (bubble = spread of points)",
        labels={"mean_pts_lost": "avg points lost (max - mean)",
                "top10_vs_rest": "top-10 points advantage over rest"},
    ).update_traces(textposition="top center").update_layout(showlegend=False)


def box_fig(data, top_k):
    d = data.copy()
    d["group"] = np.where(
        d.groupby("competition")["overall"].rank(ascending=False) <= top_k,
        f"top {top_k}", "rest")
    cols = [f"{e}_pct" for e in available_events(d)]
    long = d.melt(id_vars="group", value_vars=cols, var_name="event",
                  value_name="percentile").dropna()
    long["event"] = long["event"].str[:-4]
    return px.box(long, x="event", y="percentile", color="group",
                  title="Event percentile distribution: top finishers vs the rest")


def regression_fig(beta):
    b = beta.reset_index(names="event")
    return px.bar(b, x="event", y="std_beta",
                  title=f"Unique contribution to overall (standardised beta, "
                        f"R² = {beta.attrs['r2']:.2f}, n = {beta.attrs['n']})")


# --------------------------------------------------------------------------
# 6. Tkinter GUI
# --------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Baja SAE Insights")
        self.geometry("1100x720")
        self.raw = None          # all competitions, normalised
        self.data = None         # filtered by competition selector
        self.all_max = {}
        self.results = {}
        self.last_fig = None

        self._build_toolbar()
        self._build_tabs()
        self._build_plot_panel()
        self.status = tk.StringVar(value="Open a results workbook (.xlsx, one sheet per competition) to begin.")
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w").pack(
            fill="x", side="bottom")

    # ------------------------------------------------------------------ UI
    def _build_toolbar(self):
        bar = ttk.Frame(self, padding=6)
        bar.pack(fill="x")
        ttk.Button(bar, text="Open workbook…", command=self.open_workbook).pack(side="left")
        ttk.Button(bar, text="Scrape URLs…", command=self.scrape_dialog).pack(side="left", padx=4)

        ttk.Label(bar, text="Competition:").pack(side="left", padx=(16, 2))
        self.comp_var = tk.StringVar(value="All")
        self.comp_box = ttk.Combobox(bar, textvariable=self.comp_var, state="readonly",
                                     values=["All"], width=22)
        self.comp_box.pack(side="left")
        self.comp_box.bind("<<ComboboxSelected>>", lambda e: self.recompute())

        ttk.Label(bar, text="Top-K:").pack(side="left", padx=(16, 2))
        self.topk = tk.IntVar(value=10)
        ttk.Spinbox(bar, from_=3, to=30, width=4, textvariable=self.topk,
                    command=self.recompute).pack(side="left")

        self.keep_zeros = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Keep 0-point scores", variable=self.keep_zeros,
                        command=self.reload).pack(side="left", padx=16)

    def _build_tabs(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=6)
        self.trees = {}
        for key, label in [("pairs", "Event pairs"), ("drivers", "Overall drivers"),
                           ("gap", "Top vs field"), ("lev", "Points leverage"),
                           ("beta", "Unique contribution")]:
            frame = ttk.Frame(self.nb)
            tree = ttk.Treeview(frame, show="headings")
            vs = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=vs.set)
            tree.pack(side="left", fill="both", expand=True)
            vs.pack(side="right", fill="y")
            self.nb.add(frame, text=label)
            self.trees[key] = tree

    def _build_plot_panel(self):
        p = ttk.LabelFrame(self, text="Plotly charts (open in your browser)", padding=6)
        p.pack(fill="x", padx=6, pady=6)

        row1 = ttk.Frame(p)
        row1.pack(fill="x")
        for text, fn in [("Correlation heatmap", lambda: self.show(lambda: heatmap_fig(self.data))),
                         ("Overall drivers", lambda: self.show(lambda: drivers_fig(self.results["drivers"]))),
                         ("Top vs field", lambda: self.show(lambda: gap_fig(self.results["gap"], self.topk.get()))),
                         ("Percentile boxes", lambda: self.show(lambda: box_fig(self.data, self.topk.get()))),
                         ("Points leverage", lambda: self.show(lambda: leverage_fig(self.results["lev"]))),
                         ("Unique contribution", lambda: self.show(lambda: regression_fig(self.results["beta"])))]:
            ttk.Button(row1, text=text, command=fn).pack(side="left", padx=2)

        row2 = ttk.Frame(p)
        row2.pack(fill="x", pady=(6, 0))
        ttk.Label(row2, text="Scatter:  X").pack(side="left")
        self.x_var, self.y_var = tk.StringVar(), tk.StringVar()
        self.x_box = ttk.Combobox(row2, textvariable=self.x_var, state="readonly", width=14)
        self.y_box = ttk.Combobox(row2, textvariable=self.y_var, state="readonly", width=14)
        self.x_box.pack(side="left", padx=4)
        ttk.Label(row2, text="Y").pack(side="left")
        self.y_box.pack(side="left", padx=4)
        self.mode_var = tk.StringVar(value="Percentile")
        ttk.Combobox(row2, textvariable=self.mode_var, state="readonly", width=10,
                     values=["Percentile", "Points"]).pack(side="left", padx=4)
        ttk.Button(row2, text="Plot scatter",
                   command=lambda: self.show(lambda: scatter_fig(
                       self.data, self.x_var.get(), self.y_var.get(), self.mode_var.get()))
                   ).pack(side="left", padx=4)
        ttk.Button(row2, text="Save last chart as HTML…", command=self.save_html).pack(side="right")

    # ------------------------------------------------------------ data flow
    def open_workbook(self):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx *.xlsm *.xls")])
        if path:
            self.path = path
            self.reload()

    def reload(self):
        if not getattr(self, "path", None):
            return
        try:
            self.config(cursor="watch")
            self.update_idletasks()
            data, self.all_max = load_workbook(self.path, not self.keep_zeros.get())
            self.raw = data
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Could not load workbook", str(e))
            return
        finally:
            self.config(cursor="")
        um = self.raw.attrs.get("unmatched", {})
        if um:
            messagebox.showwarning(
                "Unrecognised columns",
                "These columns were not matched to any known event and were ignored:\n\n"
                + "\n".join(f"{k}: {', '.join(v)}" for k, v in um.items())
                + "\n\nAdd a pattern to COLUMN_PATTERNS if any are real events.")
        comps = sorted(self.raw["competition"].unique())
        self.comp_box["values"] = ["All"] + comps
        if self.comp_var.get() not in self.comp_box["values"]:
            self.comp_var.set("All")
        self.recompute()

    def recompute(self):
        if self.raw is None:
            return
        sel = self.comp_var.get()
        df = self.raw if sel == "All" else self.raw[self.raw["competition"] == sel]
        self.data = add_normalised(df.reset_index(drop=True))
        evs = available_events(self.data, min_n=10)
        if len(evs) < 2:
            messagebox.showwarning("Not enough data", "Fewer than two events have usable scores.")
            return
        k = max(3, int(self.topk.get()))
        r = self.results
        r["pairs"] = pairwise_table(self.data, evs, min_n=10)
        r["drivers"] = overall_drivers(self.data, min_n=10)
        r["gap"] = top_vs_field(self.data, k)
        r["lev"] = points_leverage(self.data, self.all_max)
        r["beta"] = standardized_regression(self.data)

        self._fill("pairs", r["pairs"])
        self._fill("drivers", r["drivers"])
        self._fill("gap", r["gap"].reset_index(names="event"))
        self._fill("lev", r["lev"])
        self._fill("beta", r["beta"].reset_index(names="event") if r["beta"] is not None
                   else pd.DataFrame({"note": ["Not enough complete rows for regression"]}))

        for box in (self.x_box, self.y_box):
            box["values"] = evs + ["overall"]
        defaults = ("hill_climb", "accel") if {"hill_climb", "accel"} <= set(evs) else (evs[0], evs[1])
        self.x_var.set(defaults[0])
        self.y_var.set(defaults[1])
        self.status.set(f"{sel}: {len(self.data)} team results, "
                        f"{self.data['competition'].nunique()} competition(s), "
                        f"events: {', '.join(evs)}")

    def _fill(self, key, df):
        tree = self.trees[key]
        tree.delete(*tree.get_children())
        tree["columns"] = list(df.columns)
        for c in df.columns:
            tree.heading(c, text=c)
            tree.column(c, width=110, anchor="center")
        for _, row in df.iterrows():
            tree.insert("", "end", values=[f"{v:.2f}" if isinstance(v, float) else v
                                           for v in row])

    # --------------------------------------------------------------- charts
    def show(self, builder):
        if self.data is None:
            messagebox.showinfo("No data", "Open a workbook first.")
            return
        try:
            fig = builder()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Chart error", str(e))
            return
        if fig is None:
            messagebox.showinfo("Nothing to plot", "Not enough data for this chart.")
            return
        self.last_fig = fig
        fig.show()   # opens in the default browser

    def save_html(self):
        if self.last_fig is None:
            messagebox.showinfo("No chart", "Make a chart first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".html",
                                            filetypes=[("HTML", "*.html")])
        if path:
            self.last_fig.write_html(path)

    # -------------------------------------------------------------- scraping
    def scrape_dialog(self):
        win = tk.Toplevel(self)
        win.title("Scrape results")
        ttk.Label(win, text="One URL per line (results page, or direct .xlsx/.csv link).\n"
                            "Optional: put a sheet name first, e.g.  Ohio 26 | https://…"
                  ).pack(padx=8, pady=6)
        txt = tk.Text(win, width=90, height=10)
        txt.pack(padx=8)

        def go():
            lines = [l.strip() for l in txt.get("1.0", "end").splitlines() if l.strip()]
            if not lines:
                return
            names, urls = [], []
            for l in lines:
                if "|" in l:
                    n, u = [s.strip() for s in l.split("|", 1)]
                else:
                    n, u = None, l
                names.append(n)
                urls.append(u)
            out = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                               filetypes=[("Excel", "*.xlsx")],
                                               title="Save scraped workbook as")
            if not out:
                return
            win.destroy()
            self.status.set("Scraping…")

            def work():
                try:
                    scrape_to_excel(urls, out, names)
                    self.after(0, lambda: self._scrape_done(out))
                except Exception as e:  # noqa: BLE001
                    self.after(0, lambda: messagebox.showerror("Scrape failed", str(e)))
                    self.after(0, lambda: self.status.set("Scrape failed."))

            threading.Thread(target=work, daemon=True).start()

        ttk.Button(win, text="Scrape & load", command=go).pack(pady=8)

    def _scrape_done(self, out):
        self.path = out
        self.reload()


if __name__ == "__main__":
    App().mainloop()
