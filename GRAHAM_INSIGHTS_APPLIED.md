# Ian Graham "How to Win the Premier League" — Applied to SC Draft Analytics

> Direct learnings from the full text, translated into specific, implementable ideas.
> Priority: **HIGH** = implement next sprint | **MEDIUM** = next model version | **LOW** = research/exploratory

---

## 1. Bayesian Averaging: 2-Game Sample Should Barely Move the Prior

**From the book:** Graham's team strength update rule: "Your team's strength tomorrow is about 98% of your team's strength yesterday, plus 2% of how well you did compared to expectations today." Derived from Bayesian updating where goals (rare events) provide weak evidence per game. Concrete case: Adam Le Fondre scores 9 goals at 0.54/90 in one season (small sample) → Bayesian revision to 0.40. Diego Costa scores 52 goals at 0.53/90 over three seasons → barely moves from 0.49. Same rate, completely different reliability. Minimum threshold for reliable ratings: ~2,500 minutes (~28 AFL games at 90 mins each). Giovani dos Santos on 1,400 minutes (~15 games): "It is easier to look excellent in 15 games than in 30 games."

**Applied here:** After R1 and R2 (2 games), our `recent3_avg` gets 40% weight in `pick_score`. With only 2 games this is far too aggressive — the posterior should be almost identical to the prior.

**Specific implementation** (Priority: HIGH):

Replace the fixed 40/60 blend in `waiver/newsletter.py` with a sample-size-weighted Bayesian posterior:

```python
# In waiver/metrics.py, add to score_all_players():
def bayesian_avg(row) -> float:
    """
    Posterior mean = (n * observed + k * prior) / (n + k)
    k = "equivalent prior games" — how many games the historical avg is worth.
    Calibrated: 20 games prior weight means 2 current games move it ~9%.
    """
    n = len(row.get("scores_2026") or [])
    prior = row["effective_avg"]          # weighted historical avg
    if n == 0:
        return prior
    observed = sum(row["scores_2026"]) / n
    k = 20  # prior equivalent sample size
    return (n * observed + k * prior) / (n + k)

df["bayesian_avg"] = df.apply(bayesian_avg, axis=1)
```

Replace `pick_score` blending with `bayesian_avg` directly. At R2 (n=2): weight on current season = 2/(2+20) = 9%. At R10 (n=10): weight = 33%. At R18 (n=18): weight = 47%. This is mathematically correct and prevents over-reacting to hot starts.

---

## 2. Player Sub-Type Clustering (The Firmino Insight) — The Biggest Untapped Opportunity

**From the book:** Graham found Firmino was being evaluated as an ordinary striker. When he separated Firmino's games *by role* (striker vs. attacking midfielder), Firmino rated as one of the best young number 10s in Europe — above Sánchez, Rodríguez, Isco, Pogba, Ramsey. The market was comparing him to the *wrong peer group*. Graham also ran PCA on ~70 event variables and found players cluster into natural subtypes: 2 types of CB, 4 types of midfielder (destroyer, direct passer, all-rounder, number 10), 3 types of striker. **Target men almost never successfully convert to a different style after a transfer.**

**Applied here:** Our TPOR uses a single replacement level per position (DEF/MID/FWD/RUC). But "MID" contains at minimum:
- **Inside mid**: contested, clearance-dominant, high tackles, low efficiency (e.g. Bontempelli, Cripps)
- **Outside mid / wing**: high disposal, high efficiency, low contested, high metres gained (e.g. Brayshaw, Ward)
- **MID/FWD hybrid**: goal-oriented mid-forward (e.g. Rachele)

A MID replacement level of 103.1 penalises inside mids unfairly (they score lower but there are fewer elite options) and undervalues wings. The same logic applies to DEF (lockdown backmen vs. rebounding DEFs) and FWD (key position vs. small forward).

**Specific implementation** (Priority: MEDIUM):

```python
# In feature_engineering.py, add sub-type classification:
def classify_mid_subtype(row) -> str:
    """Classify midfielders into inside/outside/hybrid using ratio features."""
    contested_rate = row.get("Contested Possessions", 0) / max(row.get("disposals", 1), 1)
    clearance_rate = row.get("Clearances", 0) / max(row.get("disposals", 1), 1)
    efficiency = row.get("Disposal efficiency", 0.7)
    metres_per_disposal = row.get("Metres gained", 0) / max(row.get("disposals", 1), 1)

    if contested_rate > 0.45 or clearance_rate > 0.12:
        return "MID_INSIDE"
    elif efficiency > 0.75 and metres_per_disposal > 8:
        return "MID_OUTSIDE"
    else:
        return "MID_GENERAL"
```

Then compute separate replacement levels per sub-type and use sub-type TPOR in the newsletter. This is analogous to Graham's positional-level benchmarking and will dramatically improve signal quality for MID recommendations.

---

## 3. Context-Adjusted Scores — Strip Team Quality from Individual Ratings

**From the book:** Robertson at Hull looked like a mediocre left-back to conventional scouts because Hull were being relegated. Graham's possession value model "strips team context — it measures what the player contributed to goal probability on each action, not what the team did around him." Result: Robertson looked world-class in the data despite zero competition from top-6 clubs. Conversely, the "Ridgewell Problem": a Birmingham defender appeared elite in event data because the model gave him credit for interventions at dangerous locations, without penalising him for the defensive dysfunction that *created* those dangerous locations. Fix: allocate "defensive debits" for conceded chances based on each player's positional responsibility.

**Applied here:** SC scores are naturally affected by team quality. A midfielder on a strong AFL club (e.g. Geelong, Brisbane) has more quality teammates creating space, more disposals available, and plays against weaker defensive opponents more often. Our effective_avg treats a 95 average at Carlton (rebuilding) the same as a 95 average at Brisbane (top-4). The Carlton player is almost certainly undervalued.

**Specific implementation** (Priority: MEDIUM):

Build a per-team "SC environment adjustment" from the fanfooty historical data:

```python
# In build_features.py:
def compute_team_sc_environment(df: pd.DataFrame) -> pd.Series:
    """
    For each team-year combination, compute the average SC score of all players
    on that team. Difference from league-wide average = team_sc_boost.
    A player at a weak team has a negative boost environment → their raw avg
    understates their true quality.
    """
    team_year_avg = (
        df.groupby(["Team", "Year"])["SC"]
        .mean()
        .reset_index(name="team_avg_sc")
    )
    league_avg = df.groupby("Year")["SC"].mean().reset_index(name="league_avg")
    merged = team_year_avg.merge(league_avg, on="Year")
    merged["team_sc_boost"] = merged["team_avg_sc"] - merged["league_avg"]
    return merged

# Then: adjusted_avg = effective_avg - team_sc_boost * 0.5
# (0.5 because team quality is partially the player's own contribution)
```

Surface this in the newsletter as a "team-adjusted avg" column, or use it as a tiebreaker. Players on low-ranked teams whose team_sc_boost is strongly negative are systematically under-priced.

---

## 4. "Dangerous Disposals" vs. Safe Recycling — AFL's Equivalent of Pass Completion Being Meaningless

**From the book:** "Players who attempt dangerous passes can get away with a lower completion rate: fortune favours the brave." High pass completion can mask safe backward recycling that adds almost zero goal probability. Graham's GPA model values passes by their *goal probability contribution*, not their completion rate. The passes he prizes: "passes that go behind the opposition defence and take four or five defenders out of the game" — even at 50% completion, these are net positive. He created "Dangerous Possession Dominance" (possession advantage in attacking third only), which correlated far more strongly with success than overall possession. Wigan had plenty of possession, zero attacking-third dominance, negative goal difference.

**Applied here:** Our features include disposal efficiency, but a player with 80% efficiency doing safe handballs to a stationary teammate is rated equally to one with 80% efficiency hitting running leads through traffic. The fanfooty data has `Metres gained`, `Clearances`, and `Contested Possessions` — these are the "dangerous" AFL stats.

**Specific implementation** (Priority: HIGH, uses existing data):

Add a `danger_rate` feature to `feature_engineering.py`:

```python
def compute_danger_rate(row) -> float:
    """
    Proportion of a player's actions that are 'dangerous' —
    moving the ball forward under pressure toward goal.

    Equivalent to Graham's 'attacking-third possession dominance':
    not total volume but high-value volume.
    """
    disposals = row.get("Kicks", 0) + row.get("Handballs", 0)
    if disposals < 5:
        return 0.0

    high_value_actions = (
        row.get("Clearances", 0) * 3.0          # generates scoring chains
        + row.get("Contested Possessions", 0) * 1.5   # won under pressure
        + row.get("Goals", 0) * 6.0             # direct scoring
        + row.get("Metres gained", 0) / 15.0    # territory gain
    )
    return high_value_actions / disposals

df["danger_rate"] = df.apply(compute_danger_rate, axis=1)
```

Add `danger_rate` as a feature in `train_model.py`'s feature list. Also surface it in the newsletter as context for why a player ranks where they do — a low-SC player with high `danger_rate` is doing more work than the scoreline suggests (the Robertson analogy).

---

## 5. Hot Streaks = Regression Bait — Flag in Newsletter

**From the book:** Markovic was "the hottest young prospect in scouting departments across Europe in 2013/14." Graham: "The games where Marković was scouted happened to be his best performances of the season." Multiple clubs bid simultaneously, inflating the price beyond any analytical justification. Similarly: "Clubs tend to bid for players on scoring streaks, winning runs, or impressive World Cups. These streaks typically do not continue." The book explicitly shows that seasonal overperformance does not predict next-season overperformance — 34 of 69 Bundesliga cases, above-expectation teams *fell below* expectation the following season.

**Applied here:** In rounds 1-2 the top scorers (Dunkley 290, Macrae 289, Zorko 275) will attract pickup attention from all 8 coaches. Our model currently ranks them highly on `recent3_avg`. This is exactly the Marković trap — we're amplifying the signal from their 2 best games when they may have been scouted on their outlier games.

**Specific implementation** (Priority: HIGH):

Add a regression warning to the newsletter for any free agent where `recent3_avg > effective_avg * 1.35`:

```python
# In waiver/newsletter.py, in the row-building loop:
def _regression_warning(row) -> str:
    recent = row.get("recent3_avg") or 0
    hist = row.get("effective_avg") or 1
    if recent > 0 and hist > 0 and recent / hist > 1.35:
        return f"⚠️ hot streak (+{recent - hist:.0f} above avg)"
    return ""
```

Show this in a "Notes" column on the positional targets table. This is the exact corrective Graham recommends: surface the signal so the analyst (you) can apply human judgment.

---

## 6. Durability as an Undervalued Asset in Waiver Pickups

**From the book:** Firmino had "played nearly every game for Hoffenheim over the past four years" — this was a positive signal the market wasn't pricing. Graham: "Durability is almost as important as quality. Messi and Ronaldo's career dominance was substantially enabled by playing 3,800–3,900 top-level minutes per season for nearly 20 years. Availability is almost as important as quality." Keïta had limited injury history at Leipzig but broke down repeatedly at Liverpool, making him one of Klopp's few analytical failures.

**Applied here:** In an 8-team league where you have 23 players and must field Best 19, a free agent who plays 18/22 rounds is worth dramatically more than one who averages 120 in the games he plays but misses 6 rounds. Our `games_played_last_season` feature exists but is not surfaced in the newsletter or heavily weighted in `pick_score`.

**Specific implementation** (Priority: HIGH, uses existing data):

Add a `durability_score` metric in `waiver/metrics.py`:

```python
def compute_durability(scores_by_year: dict) -> float:
    """
    % of available rounds played across last 2 seasons.
    More reliable than single-season games played.
    Penalises players who miss 4+ rounds per season heavily.
    """
    total_played = 0
    total_available = 0
    for year in [2025, 2024]:
        games = scores_by_year.get(year, [])
        total_played += len([s for s in games if s > 0])
        total_available += 22  # standard AFL season
    if total_available == 0:
        return 0.5  # unknown
    return total_played / total_available

df["durability_score"] = df.apply(lambda r: compute_durability(r.get("scores_by_year", {})), axis=1)
```

Modify `pick_score` to incorporate durability: `pick_score = bayesian_avg * (0.7 + 0.3 * durability_score)`. A player who plays 85% of games (18/22) gets a ×1.0 modifier. One who plays 50% (11/22) gets a ×0.85 modifier — a meaningful penalty.

Surface `durability_score` as "Avail%" in the newsletter tables (shows as "82%" etc.).

---

## 7. The Pareto Frontier Trap — Why "One Currency" Matters

**From the book:** "If you optimise across 2 metrics, 13 of 492 midfielders are 'optimal'. Across 7 metrics, 133 are optimal. Across 10, 240 are optimal — half the pool. Cherry-picking metrics can justify almost any transfer." Solution: collapse everything into one currency (GPA) before comparison. Detailed metrics are used only to *explain why* a player has a certain GPA, not to score them on a multi-metric rubric.

**Applied here:** The newsletter currently shows: Proj. Score, P(80+), P(100+), P(120+), CI%, Avg, FlexBonus, TPOR, danger_rate (new). This is 8+ metrics — the coach reading it can justify picking almost anyone. The rankings need to resolve to a *single ranked list*.

**Specific implementation** (Priority: MEDIUM):

1. Keep all metrics as *informational columns* (for human context, à la Graham's "explains why")
2. But produce a single `pick_rank` per position that is THE recommendation, computed as:

```python
# In waiver/metrics.py:
def compute_pick_rank_score(row) -> float:
    """
    Single currency ranking score. Everything else in the newsletter
    is explanation, not ranking. Inspired by Graham's GPA unification.

    Components:
    - bayesian_avg: core quality signal, Bayesian-weighted current/historical
    - durability_score: availability adjustment (Graham: 'as important as quality')
    - danger_rate_percentile: strips team context, values high-value actions
    - regression_penalty: if on a hot streak, reduce by regression factor
    """
    base = row["bayesian_avg"]
    availability_mult = 0.7 + 0.3 * row.get("durability_score", 0.8)

    # Regression adjustment: if hot streak, blend back toward long-term avg
    recent = row.get("recent3_avg") or base
    long_term = row.get("effective_avg") or base
    if recent > long_term * 1.35:
        adjusted_base = 0.7 * base + 0.3 * long_term  # regression toward mean
    else:
        adjusted_base = base

    return adjusted_base * availability_mult

df["pick_rank_score"] = df.apply(compute_pick_rank_score, axis=1)
```

Sort each positional table by `pick_rank_score`. Label the #1 pick with **★ Top Pick**. The reader still sees all the explanatory columns, but there is a single authoritative recommendation.

---

## 8. Opponent-Adjusted Scoring — Both Sides of the Ledger

**From the book:** Graham built `opponent_strength_rating` into Liverpool's model (how many goals per game the opponent concedes). His key refinement on defence: "allocate defensive responsibility for opposition chances based on each player's typical positional location. When a shot is conceded centrally, centre-backs share responsibility; when a cross comes from the left, the left-back takes primary responsibility." This means giving defensive *debits* not just offensive credits. Van Dijk rated better in tracking data than event data because the model saw him preventing opponents from reaching threatening positions — "without touching the ball, he forced a worse decision."

**Applied here:** We already have `opponent_strength_rating` in the model. But the newsletter and `pick_score` don't surface opponent quality for *current week pickups*. R3 matchups matter.

**Specific implementation** (Priority: HIGH, mostly done):

The `projected_score` already incorporates opponent strength. Ensure the "vs" column in the positional targets table is colour-coded by opponent difficulty — not just showing the team name but their SC concession rank:

```python
# In waiver/newsletter.py:
def _opp_with_difficulty(opponent: str, opp_rank: int) -> str:
    """
    Annotate opponent with their defensive difficulty rank.
    opp_rank: 1 = easiest to score against, 18 = hardest.
    """
    if opp_rank is None:
        return opponent or "—"
    if opp_rank <= 6:
        indicator = "🟢"  # easy
    elif opp_rank <= 12:
        indicator = "🟡"  # average
    else:
        indicator = "🔴"  # hard
    return f"{indicator} {opponent}"
```

This makes the fixture-adjusted recommendation visible at a glance — Graham's core "performance vs. results" principle surfaced for the user every week.

---

## 9. Young Player Uplift — "Buying the Promise"

**From the book:** "When you buy a 21-year-old, you're not just buying his performance today; you're buying the promise that he'll improve in the future." Liverpool's informal age limit: rarely sign over 24 (relaxed only for specific squad roles). Brentford's strategy: sign at 19-23, sell after peak. The ageing curve — improvement through early 20s, plateau late 20s, decline in 30s — is universal and predictable. "Players who are able to play at the highest levels when young will make the most of their careers."

**Applied here:** Free agents aged 20-24 who are trending upward should be ranked higher than their current `effective_avg` implies because their true value is the *projected* average 2 years from now. The newsletter currently treats a 23-year-old averaging 85 identically to a 31-year-old averaging 85. In a season-long league this is wrong — the 23-year-old's floor is 85 and rising.

**Specific implementation** (Priority: MEDIUM):

```python
# In waiver/metrics.py:
def age_trajectory_multiplier(age: int, trend: float) -> float:
    """
    Apply an uplift for young improving players.

    trend: (2025_avg - 2024_avg) from historical data
    Positive trend + young age = upside multiplier.

    Based on Graham's ageing curve:
    - Age 18-22: strong upside, large multiplier
    - Age 23-26: moderate upside
    - Age 27-29: peak, no multiplier
    - Age 30+: slight discount
    """
    if age is None or pd.isna(age):
        return 1.0
    age = int(age)
    if age <= 22:
        base_mult = 1.08
    elif age <= 25:
        base_mult = 1.04
    elif age <= 29:
        base_mult = 1.0
    else:
        base_mult = 0.97  # mild decline discount

    # If trending upward AND young, compound the uplift
    if trend > 5 and age <= 25:
        return base_mult * 1.03
    return base_mult

df["age_mult"] = df.apply(
    lambda r: age_trajectory_multiplier(r.get("age"), r.get("sc_trend_yoy", 0)), axis=1
)
df["pick_rank_score"] = df["pick_rank_score"] * df["age_mult"]
```

This directly implements Graham's "buy the promise" principle for waiver pickups — a rising 22-year-old is a better hold than an equivalent 31-year-old.

---

## 10. Tag Notes as a Pre-Round Multiplier (Not Just a Binary Feature)

**From the book:** Graham's weekly operational rhythm: after every match, tracking data was ingested, composition ran, and player ratings updated by Sunday night. The research team looked specifically for players who "zoomed up the rankings" week-on-week. They also maintained a 20-30 game *representative* video analysis for serious targets specifically to avoid the Marković hot-streak bias.

The Tag Notes in fanfooty are the closest AFL equivalent to Liverpool's weekly player intelligence updates. Currently they're converted to 10 binary features (hot, gun, tagger, etc.) in the model. The full text notes contain richer information ("CBA increase", "locked in as the #1 ruck", "managing load — unlikely R4", "best game in two years").

**Specific implementation** (Priority: HIGH, existing data unused):

Build a `tag_signal_score` in `waiver/role_watch.py`:

```python
POSITIVE_TAG_SIGNALS = {
    "cba": +8,           # clearance role = big scoring boost
    "increase": +5,
    "locked": +6,        # locked in role = job security
    "first ruck": +8,
    "best": +4,
    "star": +3,
    "gun": +3,
    "hot": +2,
    "job security": +5,
    "100%": +5,
    "sole ruck": +8,
}

NEGATIVE_TAG_SIGNALS = {
    "managed": -6,       # load management = likely to miss
    "sore": -8,
    "injured": -12,
    "tag": -5,           # being tagged
    "tagged": -5,
    "restricted": -4,
    "unsure": -4,
    "sub": -6,           # medical sub risk
    "omission": -3,
}

def tag_signal_score(tag_notes: str) -> int:
    if not isinstance(tag_notes, str):
        return 0
    notes_lower = tag_notes.lower()
    score = 0
    for kw, val in POSITIVE_TAG_SIGNALS.items():
        if kw in notes_lower:
            score += val
    for kw, val in NEGATIVE_TAG_SIGNALS.items():
        if kw in notes_lower:
            score += val
    return score
```

Apply `tag_signal_score` as a final additive modifier to `projected_score` before ranking. This is exactly what Graham's weekly player intelligence update fed into — not the base model, but the *current-week* overlay.

---

## 11. Overachievement Regression Warning on My Roster

**From the book:** "Over 9 Bundesliga seasons, of teams that gained more points than expected, 34 of 69 gained *fewer* than expected the next season." Graham explicitly used this to argue that a mid-table Dortmund season was *noise*, not signal — same squad could easily finish 2nd the next year. The Liverpool Suárez case: "The entire +1.0 goal difference without Suárez was driven by a single 6-0 win against Newcastle immediately after his ban. The sample was 10 games. The correct conclusion was 'no signal', not 'sell Suárez'."

**Applied here:** After R1+R2, players who massively outperformed expectations (Bailey Dale +61 vs expected) should not trigger a "extend contract" signal — it's early-season noise. Conversely, underperformers like Ward (-43 vs expected) at only 2 data points barely update the prior — the correct read is "no signal yet."

**Specific implementation** (Priority: MEDIUM):

Add a "context" note to the Drop Zone section:

```python
# In waiver/newsletter.py, _section_drop_zone():
def _bayesian_confidence(row, n_games: int) -> str:
    """Flag how confident the drop recommendation is."""
    if n_games <= 3:
        return "⚠️ Low confidence (< 3 games)"
    elif n_games <= 8:
        return "Moderate"
    else:
        return "High confidence"
```

Show this next to each player in the Drop Zone table. Prevents panic-dropping a player after 2 bad games — exactly Graham's lesson from Suárez and the Dortmund overperformance analysis.

---

## 12. The Committee Principle — Data + Scout + Manager Must Agree

**From the book:** "If data, scouts and manager all agree on a player, that player rarely fails." The Liverpool process required consensus: Edwards + Klopp + Graham all had to sign off. This prevented any single domain from overriding the others. The Rodgers era used the *same process* with the *same tools* and failed — because Rodgers never aligned with the analytical output.

**Applied here:** The newsletter's single biggest weakness is that it is a one-dimensional ranking. You (the coach) have context the model lacks: your league opponents' tendencies, which positions they're likely to pick up from, injury intel from Twitter/news. The model is Graham's data layer; you are Klopp.

**Specific implementation** (Priority: LOW — process, not code):

Add a "Coach Override Notes" section at the bottom of the newsletter where you can add a note for each player that will persist into next week's newsletter (stored in a `notes.json` file). This creates the paper trail of human intelligence layered over the model — analogous to Liverpool's "20-30 game video analysis" that sits alongside the quantitative ratings.

---

## 13. Cross-Team Calibration — Identify Systematically Underpriced Free Agents

**From the book:** Robertson at Hull was priced as a Championship player by the market because scouts failed to strip team context. Liverpool paid £8m — a fraction of his true value. The model showed he was world-class; the market showed he was mediocre. The gap = opportunity. Graham's key diagnostic: "a player shines through in the data but doesn't naturally shine through to conventional scouts."

**Applied here:** Run a comparison every week: of the top 20 players by `pick_rank_score`, how many are actually being picked up? Those who are not being targeted by other teams despite strong model ratings are the AFL equivalent of Robertson — the "hidden in plain sight" pickups.

**Specific implementation** (Priority: HIGH, leverage existing data):

Add an "Undiscovered" badge to the positional targets table for any player in the top 10 by `pick_rank_score` who has NOT appeared in another team's recent transactions (visible in `outputs/transactions.csv`):

```python
# In waiver/newsletter.py:
def _is_undiscovered(player_name: str, recent_transactions: set) -> bool:
    """Player ranks highly but hasn't been targeted by any league team."""
    return player_name not in recent_transactions

# Badge: show 💎 next to player name if undiscovered + top_10_rank
```

This is the most actionable direct application of Graham's "Robertson was invisible" insight — surfaces the underpriced players that conventional coaches are ignoring.

---

## Summary: Priority Matrix

| # | Insight | Implementation | Priority | Effort |
|---|---------|---------------|----------|--------|
| 1 | Bayesian avg (not fixed 60/40) | `metrics.py` | **HIGH** | Low |
| 5 | Hot streak regression warning | `newsletter.py` | **HIGH** | Low |
| 6 | Durability score in pick_rank | `metrics.py` | **HIGH** | Low |
| 4 | Danger rate feature | `feature_engineering.py` | **HIGH** | Low |
| 8 | Opponent difficulty colour coding | `newsletter.py` | **HIGH** | Low |
| 10 | Tag signal score multiplier | `role_watch.py` | **HIGH** | Medium |
| 13 | "Undiscovered" gem badge | `newsletter.py` | **HIGH** | Low |
| 3 | Team-adjusted SC environment | `build_features.py` | MEDIUM | Medium |
| 2 | Player sub-type clustering (PCA) | `feature_engineering.py` | MEDIUM | High |
| 7 | Single pick_rank_score currency | `metrics.py` | MEDIUM | Medium |
| 9 | Age trajectory multiplier | `metrics.py` | MEDIUM | Low |
| 11 | Bayesian confidence on Drop Zone | `newsletter.py` | MEDIUM | Low |
| 12 | Coach override notes persistence | `newsletter.py` | LOW | Medium |

---

*Generated from direct reading of full book text (683KB epub extracted to txt).*
*All code snippets are approximate — adapt to actual function signatures in each file.*
