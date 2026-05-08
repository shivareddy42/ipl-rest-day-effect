# IPL Rest Day Effect

**Does a longer gap between matches actually translate into IPL wins?**

I had a theory. Cricket is a heavily prepared game. Coaches study opposition batters' weak zones, bowlers' release angles, and matchups down to specific overs. Logically, a team that gets four days off before a match should be sharper than a team coming off a two day turnaround. That extra prep time has to convert into wins, right?

I ran the numbers across every IPL match from 2008 to 2026. The answer turned out to be more interesting than I expected.

## TL;DR

When two teams faced each other with unequal rest, the more rested team won **52.3%** of the time across 878 matches. That is barely above a coin flip and not statistically significant (binomial p = 0.19). When the rest gap was three or more days, the effect **disappeared entirely** (49.4%, p = 0.94). The intuition that prep time stacks into a measurable edge is not supported by the data.

![Team1 win rate by rest day differential](plots/rest_diff_winrate.png)

## The data

Source: [ritesh-ojha/IPL-DATASET](https://github.com/ritesh-ojha/IPL-DATASET) `Match_Info.csv`, which covers IPL seasons 2008 through 2026 (1,218 matches total).

After dropping no result and abandoned matches, I had 1,193 matches with a clear winner. For each match I computed how many days each team had been off since their previous match in the same season. Matches where either team was playing their first game of the season were excluded since rest is undefined there. That left **1,101 matches** for the head to head comparison.

## Method

1. **Sort matches by date** within each season.
2. **For each team in each match**, look up their previous match in the same season and compute `days_rest = current_match_date − previous_match_date`.
3. **Merge back** so each match row has `rest_t1`, `rest_t2`, and `rest_diff = rest_t1 − rest_t2`.
4. **Bucket and aggregate**:
    * Single team view: win rate grouped by a team's own rest days.
    * Head to head view: Team1 win rate grouped by `rest_diff`.
5. **Test for significance** using a two sided binomial test against the null of 50%.

Reproduction is one command (see [Reproduce](#reproduce) below).

## Findings

### 1. Single team view: win rate is essentially flat

| Days since last match | N    | Win %  |
| --------------------- | ---: | -----: |
| 1 to 2                | 728  | 51.2%  |
| 3                     | 658  | 48.0%  |
| 4                     | 392  | 49.2%  |
| 5                     | 252  | 49.6%  |
| 6                     | 125  | **56.0%** |
| 7 to 9                | 42   | 50.0%  |
| 10+                   | 23   | 52.2%  |

No monotonic trend. The 6 day bucket pops up at 56% but with only 125 matches that bump is well within noise. Across roughly 2,200 team match observations, total wins land at exactly 50%, which is the sanity check: every match has one winner and one loser.

![Single team win rate by own rest days](plots/single_team_winrate.png)

### 2. Head to head view: rest advantage is weak and non monotonic

| Team1 rest minus Team2 rest | N    | Team1 Win % |
| --------------------------- | ---: | ----------: |
| −4 or more                  | 36   | 58.3%       |
| −3                          | 47   | 48.9%       |
| −2                          | 90   | 47.8%       |
| −1                          | 279  | 46.2%       |
| 0 (same rest)               | 223  | 52.5%       |
| +1                          | 240  | 52.1%       |
| +2                          | 101  | 53.5%       |
| +3                          | 61   | 59.0%       |
| +4 or more                  | 24   | **33.3%**   |

If the hypothesis were correct, this column should slope upward as you move from `−4` to `+4`. It does, weakly, in the middle range. But the tails do the opposite of what the theory predicts. Teams with **four or more extra days of rest won only 33% of the time**.

### 3. Aggregated tests

| Comparison                          | N    | More rested team won | Binomial p |
| ----------------------------------- | ---: | -------------------: | ---------: |
| Any rest gap (1+ days)              | 878  | 459 (52.3%)          | 0.19       |
| Big rest gap (3+ days)              | 168  | 83 (49.4%)           | 0.94       |

## Why the hypothesis fails

A few things are likely happening at once.

**Floor effect.** Even two days is enough for IPL analyst teams to prepare opposition dossiers. The marginal value of day four over day two is small. Modern T20 prep is heavily pre computed: opposition matchup heatmaps, phase wise scoring rates, bowler vs handedness splits. None of that needs a week to build.

**Rust offsets prep beyond a point.** Above five or six days off, lack of match rhythm starts to hurt more than extra prep helps. T20 batting is timing intensive, and you lose feel quickly.

**Selection bias on the long tail.** A team with four or more extra days of rest is rarely a team that "chose" to prepare deeply. They are usually the team with weird circumstances: travel disruption, players away on national duty, late season scheduling for an already eliminated side, or a venue switch. Those circumstances correlate with worse outcomes for reasons that have nothing to do with rest.

**T20 noise floor is high.** With around 120 balls per side, single overs swing matches. Rest edge effects, even if real and worth a percentage point or two, get drowned in the variance of any individual match. NFL studies that found small rest effects used 20+ years of 270 game seasons. The IPL with 60 to 70 matches per year does not have that statistical power.

## Implication

For anyone modeling IPL outcomes, **rest day differential is a weak feature on its own**. Don't expect it to carry a model. If you want to use it, interact it with other variables: travel distance from the previous venue, whether key players were on national duty, whether the opponent is in must win mode, dew conditions, and so on. The conditional effect might exist even when the marginal one does not.

## Limitations

* **Confounded variable.** Long rest gaps correlate with travel patterns, injuries, and team status. This analysis does not attempt to isolate rest from those.
* **Within season only.** Rest is computed within a season, so the first match for each team is excluded. The 1,101 match base is roughly 92% of clean matches.
* **No team strength control.** A simple win rate comparison does not adjust for the fact that better teams might systematically end up with different rest patterns. A logistic regression with team strength priors would tighten the estimate.
* **No segmentation by phase.** Effects might differ between the league stage and playoffs, or between Mumbai based teams and outstation venues. Not explored here.

## Reproduce

```bash
git clone https://github.com/shivareddy42/ipl-rest-day-effect.git
cd ipl-rest-day-effect
pip install -r requirements.txt
python src/analyze.py
```

Outputs:

* Console tables for both views.
* `plots/rest_diff_winrate.png` and `plots/single_team_winrate.png`.
* `data/match_level_results.csv` with one row per match including `rest_t1`, `rest_t2`, `rest_diff`, and `t1_won`.

## Repo layout

```
ipl-rest-day-effect/
├── README.md
├── requirements.txt
├── LICENSE
├── data/
│   ├── Match_Info.csv              # source data, redistributed from ritesh-ojha/IPL-DATASET
│   └── match_level_results.csv     # output: per match rest features and outcomes
├── plots/
│   ├── rest_diff_winrate.png
│   └── single_team_winrate.png
└── src/
    └── analyze.py
```

## Credits

Match data: [ritesh-ojha/IPL-DATASET](https://github.com/ritesh-ojha/IPL-DATASET).

If you extend this with travel distance, dew conditions, or playoff splits, send me the link. I want to see what the conditional effects look like.

## License

MIT
