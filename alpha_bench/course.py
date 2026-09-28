"""Ground truth from WorldQuant's Learn2Quant lesson series (YouTube @WorldQuantCareers).

Transcripts were read in full and the on-screen BRAIN screens (expressions, settings dialogs, IS
summaries) were read from the videos' timeline thumbnails. Numbers below are what BRAIN displayed in
the lessons. Where a thumbnail was too blurry to be certain, the field is marked `approx`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

SOURCE = "Learn2Quant (WorldQuant, youtube.com/@WorldQuantCareers)"


@dataclass
class LessonAlpha:
    id: str
    lesson: str
    title: str
    idea: str
    expression: str
    settings: dict
    brain: dict                     # IS summary shown on screen
    verdict: str                    # what BRAIN's criteria make of it
    improvement: str                # the lesson's own suggestion for improving it
    video: str
    approx: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def pasted_text(self) -> str:
        """The IS summary as BRAIN shows it, for the Reconcile tab."""
        b = self.brain
        cols = [("Sharpe", "sharpe", "{:.2f}"), ("Turnover", "turnover", "{:.2%}"), ("Fitness", "fitness", "{:.2f}"),
                ("Returns", "returns", "{:.2%}"), ("Drawdown", "drawdown", "{:.2%}"), ("Margin", "margin", "{:.2f}‱")]
        present = [(h, k, f) for h, k, f in cols if b.get(k) is not None]
        return ("  ".join(h for h, _, _ in present) + "\n" +
                "  ".join(f.format(b[k]) for _, k, f in present))


EXAMPLES: list[LessonAlpha] = [
    LessonAlpha(
        "l4-open-close", "Lesson 4", "Open-to-close reversion within sub-industry",
        "If a stock closes below its open, expect it to bounce back relative to peers (price-volume reversion).",
        "-group_rank(close - open, subindustry)",
        {"region": "USA", "universe": "TOP3000", "delay": 1, "decay": 10, "neutralization": "SUBINDUSTRY",
         "truncation": 0.01},
        {"sharpe": 1.70, "turnover": 0.578, "fitness": 0.71, "returns": 0.1008, "drawdown": 0.0495, "margin": 3.46},
        "Fails: fitness 0.71 is below 1.0 because turnover (~58%) is high.",
        "Control turnover by trading only in conditions that favour reversion: trade in higher-volatility "
        "conditions and hold during low volatility.",
        "https://www.youtube.com/watch?v=40JActnyhkM&t=164", approx=["turnover"]),
    LessonAlpha(
        "l4-cashflow-yield", "Lesson 4", "Improving cash-flow yield",
        "If cash flow from operations relative to market cap is improving, the stock should outperform.",
        "ts_zscore(cashflow_op / cap, 60)",
        {"region": "USA", "universe": "TOP3000", "delay": 1, "decay": 0, "neutralization": "MARKET",
         "truncation": 0.01},
        {"sharpe": 2.00, "turnover": 0.29, "fitness": 1.09, "returns": 0.09, "drawdown": 0.0391, "margin": 5.72},
        "Passes the delay-1 Sharpe (>= 1.25) and fitness (>= 1.0) floors.",
        "Use analyst estimates of operating cash flow instead of the last reported value, since they arrive "
        "before quarterly reports.",
        "https://www.youtube.com/watch?v=40JActnyhkM&t=400", approx=["decay", "truncation"]),
    LessonAlpha(
        "l5-buzz", "Lesson 5", "Sentiment buzz reversal",
        "Rising sentiment buzz versus its recent average means the stock is attracting attention and may be "
        "overpriced: short it; go long falling buzz.",
        "window = 10;\n-1 * snt_buzz / ts_mean(snt_buzz, window)",
        {"region": "USA", "universe": "TOP200", "delay": 1, "decay": 10, "neutralization": "INDUSTRY",
         "truncation": 0.01},
        {"sharpe": 1.60, "turnover": 0.4407, "fitness": 0.75, "returns": 0.0975, "drawdown": 0.0466, "margin": 4.42},
        "Fails: fitness 0.75 is below 1.0 (turnover 44%).",
        "(None given in the lesson.) Alpha Bench suggests more decay or ranking within size buckets.",
        "https://www.youtube.com/watch?v=dpNuRCcxjwc&t=123"),
    LessonAlpha(
        "l5-iv-spread", "Lesson 5", "Call-minus-put implied volatility (long horizon)",
        "If implied volatility of calls expiring up to 2 years out exceeds that of puts, demand for the stock "
        "exceeds supply: go long; otherwise short.",
        "IV_call = implied_volatility_call_720;\nIV_put = implied_volatility_put_720;\nIV_call - IV_put",
        {"region": "USA", "universe": "TOP3000", "delay": 1, "decay": 7, "neutralization": "INDUSTRY",
         "truncation": 0.01},
        {"sharpe": 1.94, "turnover": 0.25, "fitness": 1.44, "returns": 0.14, "drawdown": 0.03, "margin": 11.06},
        "Passes the delay-1 Sharpe and fitness floors.",
        "Add trends in other options variables such as open interest and options volume.",
        "https://www.youtube.com/watch?v=dpNuRCcxjwc&t=353"),
    LessonAlpha(
        "l6-volume-momentum", "Lesson 6", "Volume momentum",
        "Last week's volume well above the past year's average signals rising liquidity and bullish interest: "
        "go long.",
        "sqrt(rank(ts_mean(volume, 5) / ts_mean(volume, 240)))",
        {"region": "USA", "universe": "TOP3000", "delay": 1},
        {"sharpe": 1.50, "turnover": 0.20, "fitness": 0.84, "returns": 0.06, "drawdown": 0.048, "margin": 6.14},
        "Fails: fitness 0.84 is below 1.0 (returns only 6%). Returns/drawdown is 1.25 (> 1, good).",
        "(None given.) Alpha Bench suggests coarser neutralization or combining with a complementary signal.",
        "https://www.youtube.com/watch?v=LC6whEo80T0&t=264", approx=["settings"]),
    LessonAlpha(
        "l7-iv-parkinson", "Lesson 7", "Call implied vs realised (Parkinson) volatility, delay-0",
        "Call option demand, measured by at-the-money call implied volatility (up to 4 months) scaled by "
        "Parkinson high-low volatility over 4 months, predicts higher returns.",
        "implied_volatility_call_120 / parkinson_volatility_120",
        {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "SUBINDUSTRY"},
        {"sharpe": 2.0, "turnover": 0.29, "returns": 0.14, "drawdown": 0.1131},
        "Delay-0: Sharpe 2.0 sits right at the delay-0 floor; returns/drawdown about 1.2.",
        "(Illustrates delay-0: more timely data, lower trading capacity.)",
        "https://www.youtube.com/watch?v=XqdcIayjAug&t=336", approx=["fitness shown on screen (0.84) is "
                                                                   "inconsistent with the other figures"]),
    LessonAlpha(
        "l9-two-day-reversal", "Lesson 9", "Two-day reversal (risk-neutralization demo)",
        "Short-term price reversal, used to show risk neutralization: higher Sharpe and lower drawdown for a "
        "marginal increase in turnover.",
        "rank(-ts_delta(close, 2))",
        {"region": "USA", "universe": "TOP3000", "delay": 1},
        {"sharpe": 2.31, "turnover": 0.9403, "fitness": 0.64, "returns": 0.0745, "drawdown": 0.0344, "margin": 1.52},
        "Fails: turnover 94% is above the 70% ceiling, and fitness is about 0.64.",
        "Neutralize to common factor risks (a risk-model neutralization) to strengthen the idiosyncratic part.",
        "https://www.youtube.com/watch?v=eTq8iPhL1Ys&t=307", approx=["settings", "fitness"]),
    LessonAlpha(
        "l2-price-reversion", "Lesson 2", "Five-day price reversion",
        "Buy stocks whose price fell over the past week, short those that rose, expecting reversion.",
        "-ts_delta(close, 5)",
        {"region": "USA", "universe": "TOP3000", "delay": 1},
        {},
        "Not backtested in the lesson.",
        "",
        "https://www.youtube.com/watch?v=A3RNoYAz_9U&t=301"),
]

# Rules and guidance taught in the lessons (shown in the UI and used by the rules engine).
PRINCIPLES = [
    ("Fitness", "Fitness combines Sharpe, returns and turnover; every IS summary in the lessons matches "
                "Sharpe x sqrt(|Returns| / max(Turnover, 12.5%)).", "Lessons 4-6, 9 (IS summaries)"),
    ("Returns / drawdown", "Target a returns-to-drawdown ratio greater than 1; higher is better.", "Lesson 3, 8"),
    ("Drawdown", "Drawdown is the percentage of the largest loss incurred during any year of the backtest.",
     "Lesson 3"),
    ("Correlation", "Correlation with other alphas in your pool should be low unless the new alpha performs much "
                    "better than the correlated ones.", "Lesson 3"),
    ("Weight test", "Alpha weight must be evenly distributed across stocks, not concentrated on a few.", "Lesson 3"),
    ("Sub-universe test", "Performance in the immediate smaller set of tradable stocks must exceed a required "
                          "threshold.", "Lesson 3"),
    ("Diversity", "Combine alphas with correlation below 0.5; diversify by data category, operators "
                  "(time-series vs cross-sectional), holding period and author.", "Lesson 8"),
    ("Turnover buckets", "Low turnover < 5%, medium 5-15%, high > 15%.", "Lesson 8"),
    ("Delay", "Delay-1 uses yesterday's data (avoids look-ahead bias); delay-0 uses today's data up to a cut-off "
              "and should perform better but has less trading capacity.", "Lesson 7"),
    ("Decay", "Decay is a weighted sum of past alpha values that smooths outliers and reduces turnover.",
     "Lesson 4"),
    ("Position cap", "The worked examples restrict the maximum capital on any single stock to 1% "
                     "(truncation 0.01).", "Lessons 4, 5"),
    ("Backtest", "Simulations run over the previous 5 years; the settings dialog shows a 1-year test period.",
     "Lessons 3, 4"),
    ("Factor risk", "Neutralize to common factor risks (beta, momentum, size, value, growth, volatility, leverage, "
                    "liquidity): typically higher Sharpe and lower drawdown for slightly more turnover.", "Lesson 9"),
    ("Idea types", "Reversion, momentum, seasonality, lead-lag and index rebalancing.", "Lesson 6"),
    ("Data categories", "Price-volume, fundamental, analyst, sentiment, options, model, insider, short interest.",
     "Lesson 4"),
]


def examples() -> list[dict]:
    return [dict(e.to_dict(), pasted_text=e.pasted_text()) for e in EXAMPLES]


def by_id(example_id: str) -> LessonAlpha | None:
    return next((e for e in EXAMPLES if e.id == example_id), None)
