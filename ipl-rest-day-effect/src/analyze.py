"""
IPL Rest Day Effect: does a longer gap between matches translate into wins?

Hypothesis: more days off = more time to scout opposition batters and bowlers
            = higher win rate.

Data: ritesh-ojha/IPL-DATASET Match_Info.csv (2008 to 2026, 1,218 matches).
Method: for each match, compute days since each team's previous match within
        the same season. Compare win rates across rest day buckets and across
        rest differentials between the two teams.

Output: console tables, plots (rest_diff_winrate.png, single_team_winrate.png),
        and a per match results CSV.

Reproduce: python src/analyze.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scs

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Match_Info.csv"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

# Neon color palette to match the portfolio site aesthetic
CYAN = "#06b6d4"
PURPLE = "#a855f7"
DIM = "#475569"
BG = "#0f172a"
FG = "#e2e8f0"
GRID = "#1e293b"


def load_matches(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["match_date"]).copy()
    df["season"] = df["match_date"].dt.year

    # Keep matches with a real winner (drop "no result", missing winner)
    valid = df["winner"].notna() & (df["winner"].astype(str).str.upper() != "NA")
    df = df[valid].sort_values("match_date").reset_index(drop=True)
    return df


def compute_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Return long form (one row per team per match) with days_rest and won."""
    long = pd.concat(
        [
            df[["match_number", "match_date", "season", "team1", "winner"]]
            .rename(columns={"team1": "team"})
            .assign(side="team1"),
            df[["match_number", "match_date", "season", "team2", "winner"]]
            .rename(columns={"team2": "team"})
            .assign(side="team2"),
        ],
        ignore_index=True,
    )
    long = long.sort_values(["team", "season", "match_date"]).reset_index(drop=True)
    long["prev_date"] = long.groupby(["team", "season"])["match_date"].shift(1)
    long["days_rest"] = (long["match_date"] - long["prev_date"]).dt.days
    long["won"] = (long["team"] == long["winner"]).astype(int)
    return long


def merge_pairwise(df: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    t1 = (long[long["side"] == "team1"]
          [["match_number", "days_rest"]]
          .rename(columns={"days_rest": "rest_t1"}))
    t2 = (long[long["side"] == "team2"]
          [["match_number", "days_rest"]]
          .rename(columns={"days_rest": "rest_t2"}))
    m = df.merge(t1, on="match_number").merge(t2, on="match_number")
    m["t1_won"] = (m["winner"] == m["team1"]).astype(int)
    m["rest_diff"] = m["rest_t1"] - m["rest_t2"]
    m = m.dropna(subset=["rest_t1", "rest_t2"]).copy()
    for col in ("rest_t1", "rest_t2", "rest_diff"):
        m[col] = m[col].astype(int)
    return m


def bucket_single(d: int) -> str:
    if d <= 2: return "1 to 2"
    if d == 3: return "3"
    if d == 4: return "4"
    if d == 5: return "5"
    if d == 6: return "6"
    if d <= 9: return "7 to 9"
    return "10+"


def bucket_diff(d: int) -> str:
    if d <= -4: return "minus 4 or more"
    if d == -3: return "minus 3"
    if d == -2: return "minus 2"
    if d == -1: return "minus 1"
    if d == 0:  return "same"
    if d == 1:  return "plus 1"
    if d == 2:  return "plus 2"
    if d == 3:  return "plus 3"
    return "plus 4 or more"


def plot_rest_diff(m: pd.DataFrame, out: Path) -> None:
    order = ["minus 4 or more", "minus 3", "minus 2", "minus 1", "same",
             "plus 1", "plus 2", "plus 3", "plus 4 or more"]
    labels = ["−4+", "−3", "−2", "−1", "0", "+1", "+2", "+3", "+4+"]
    m = m.copy()
    m["bucket"] = m["rest_diff"].apply(bucket_diff)
    grouped = (m.groupby("bucket")
               .agg(n=("t1_won", "size"), wins=("t1_won", "sum"))
               .reindex(order))
    grouped["pct"] = grouped["wins"] / grouped["n"] * 100

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    bars = ax.bar(labels, grouped["pct"], color=CYAN,
                  edgecolor=PURPLE, linewidth=1.5, alpha=0.85)
    ax.axhline(50, color=PURPLE, linestyle="--", linewidth=1.2, alpha=0.6,
               label="Coin flip (50%)")
    for bar, n in zip(bars, grouped["n"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6,
                f"{h:.1f}%", ha="center", color=FG, fontsize=10, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 32.5,
                f"n={int(n)}", ha="center", color=DIM, fontsize=9)

    ax.set_xlabel("Team1 rest minus Team2 rest (days)", color=FG, fontsize=11)
    ax.set_ylabel("Team1 win rate (%)", color=FG, fontsize=11)
    ax.set_title("IPL: Team1 win rate by rest day differential\n"
                 "Seasons 2008 to 2026, 1,101 matches",
                 color=FG, fontsize=13, fontweight="bold")
    ax.set_ylim(30, 65)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.5)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    plt.tight_layout()
    plt.savefig(out, dpi=140, facecolor=BG)
    plt.close()


def plot_single_team(long: pd.DataFrame, out: Path) -> None:
    s = long.dropna(subset=["days_rest"]).copy()
    s["days_rest"] = s["days_rest"].astype(int)
    s["bucket"] = s["days_rest"].apply(bucket_single)
    order = ["1 to 2", "3", "4", "5", "6", "7 to 9", "10+"]
    grouped = (s.groupby("bucket")
               .agg(n=("won", "size"), wins=("won", "sum"))
               .reindex(order))
    grouped["pct"] = grouped["wins"] / grouped["n"] * 100

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor(BG)
    bars = ax.bar(order, grouped["pct"], color=PURPLE,
                  edgecolor=CYAN, linewidth=1.5, alpha=0.85)
    ax.axhline(50, color=CYAN, linestyle="--", linewidth=1.2, alpha=0.6,
               label="Coin flip (50%)")
    for bar, n in zip(bars, grouped["n"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                f"{h:.1f}%", ha="center", color=FG, fontsize=10, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 41,
                f"n={int(n)}", ha="center", color=DIM, fontsize=9)

    ax.set_xlabel("Days since team's previous match (same season)", color=FG, fontsize=11)
    ax.set_ylabel("Win rate (%)", color=FG, fontsize=11)
    ax.set_title("IPL: Single team win rate by own rest days\n"
                 "Seasons 2008 to 2026",
                 color=FG, fontsize=13, fontweight="bold")
    ax.set_ylim(40, 60)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.5)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    plt.tight_layout()
    plt.savefig(out, dpi=140, facecolor=BG)
    plt.close()


def report(m: pd.DataFrame, long: pd.DataFrame) -> None:
    print(f"Analysis base: {len(m)} matches with both teams having a prior match in season")
    print(f"Seasons: {sorted(m['season'].unique().tolist())}\n")

    # Single team table
    s = long.dropna(subset=["days_rest"]).copy()
    s["days_rest"] = s["days_rest"].astype(int)
    s["bucket"] = s["days_rest"].apply(bucket_single)
    print("Single team win rate by own rest days")
    print(f"{'Rest':<10} {'N':>6} {'Wins':>6} {'Win %':>8}")
    for b in ["1 to 2", "3", "4", "5", "6", "7 to 9", "10+"]:
        sub = s[s["bucket"] == b]
        n, w = len(sub), int(sub["won"].sum())
        pct = 100 * w / n if n else 0
        print(f"{b:<10} {n:>6} {w:>6} {pct:>7.2f}%")
    print()

    # Rest differential table
    m2 = m.copy()
    m2["bucket"] = m2["rest_diff"].apply(bucket_diff)
    order = ["minus 4 or more", "minus 3", "minus 2", "minus 1", "same",
             "plus 1", "plus 2", "plus 3", "plus 4 or more"]
    print("Team1 win rate by rest differential (T1 rest minus T2 rest)")
    print(f"{'Bucket':<18} {'N':>5} {'T1 Wins':>8} {'T1 Win %':>9}")
    for b in order:
        sub = m2[m2["bucket"] == b]
        n, w = len(sub), int(sub["t1_won"].sum())
        pct = 100 * w / n if n else 0
        print(f"{b:<18} {n:>5} {w:>8} {pct:>8.2f}%")
    print()

    # Aggregated framing
    rested_won = (((m["rest_diff"] > 0) & (m["t1_won"] == 1)).sum()
                  + ((m["rest_diff"] < 0) & (m["t1_won"] == 0)).sum())
    unequal = int((m["rest_diff"] != 0).sum())
    big_n = int((m["rest_diff"].abs() >= 3).sum())
    big_won = (((m["rest_diff"] >= 3) & (m["t1_won"] == 1)).sum()
               + ((m["rest_diff"] <= -3) & (m["t1_won"] == 0)).sum())
    print(f"Any rest gap: more rested team won {rested_won}/{unequal} "
          f"= {100*rested_won/unequal:.2f}%")
    print(f"Big rest gap (3+ days): more rested team won {int(big_won)}/{big_n} "
          f"= {100*big_won/big_n:.2f}%")

    # Stat tests
    p_any = scs.binomtest(int(rested_won), unequal, p=0.5).pvalue
    p_big = scs.binomtest(int(big_won), big_n, p=0.5).pvalue
    print(f"\nBinomial test, any rest gap vs 50%: p = {p_any:.4f}")
    print(f"Binomial test, 3+ day gap vs 50%:   p = {p_big:.4f}")


def main() -> None:
    df = load_matches(DATA)
    long = compute_rest_days(df)
    m = merge_pairwise(df, long)

    report(m, long)
    plot_rest_diff(m, PLOTS / "rest_diff_winrate.png")
    plot_single_team(long, PLOTS / "single_team_winrate.png")

    out_csv = ROOT / "data" / "match_level_results.csv"
    keep = ["match_date", "season", "team1", "team2", "winner",
            "rest_t1", "rest_t2", "rest_diff", "t1_won"]
    m[keep].to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")
    print(f"Wrote {PLOTS}/rest_diff_winrate.png")
    print(f"Wrote {PLOTS}/single_team_winrate.png")


if __name__ == "__main__":
    main()
