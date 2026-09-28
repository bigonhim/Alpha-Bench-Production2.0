# Alpha Bench user guide

Alpha Bench helps you produce WorldQuant BRAIN alphas that pass BRAIN's submission checks. It does three jobs:

- **Create** a new alpha from an economic idea.
- **Recreate or modify** an alpha you already have.
- **Reconcile** Alpha Bench's prediction with BRAIN's real result, and repair whatever failed.

It works in two tiers:
- A fast **offline engine** screens ideas in seconds. Its results are labelled *simulated*.
- Your **BRAIN account** gives the ground truth. Those results are labelled *verified on BRAIN*.

The rules it checks against come from WorldQuant's **Learn2Quant** lesson series plus BRAIN's published criteria. You can override every threshold (see [Rules and thresholds](#rules-and-thresholds)).

> Alpha Bench never submits anything. You always submit on the BRAIN website yourself. You're responsible for using the tool within your WorldQuant consultant agreement and BRAIN's terms of use.

---

## Contents

1. [Install and start](#install-and-start)
2. [Sign in to BRAIN](#sign-in-to-brain)
3. [The screen at a glance](#the-screen-at-a-glance)
4. [Create an alpha](#create-an-alpha)
5. [Read a result](#read-a-result)
6. [Fix a failing alpha](#fix-a-failing-alpha)
7. [Verify on BRAIN and submit](#verify-on-brain-and-submit)
8. [Recreate or modify an alpha](#recreate-or-modify-an-alpha)
9. [Reconcile with BRAIN's real results](#reconcile-with-brains-real-results)
10. [Check correlation with your other alphas](#check-correlation-with-your-other-alphas)
11. [Library and lineage](#library-and-lineage)
12. [Rules and thresholds](#rules-and-thresholds)
13. [How far to trust the offline numbers](#how-far-to-trust-the-offline-numbers)
14. [What the lessons teach, and where Alpha Bench uses it](#what-the-lessons-teach-and-where-alpha-bench-uses-it)
15. [Troubleshooting](#troubleshooting)

---

## Install and start

**Windows** (one-time setup, then start):

```powershell
cd alpha-bench
.\setup.ps1      # creates a private Python environment outside OneDrive and installs everything
.\run.ps1        # starts Alpha Bench and opens it in your browser
```

**macOS/Linux:** `python -m venv ~/.alpha-bench-venv && ~/.alpha-bench-venv/bin/pip install -e .`, then `~/.alpha-bench-venv/bin/python -m alpha_bench`.

The launcher prints a link like `http://127.0.0.1:53211/?k=…` and opens it.

- The `k=` key is new on every launch. If you open Alpha Bench in another browser, use that exact link.
- The first launch spends about 10 seconds building the offline market. A banner shows while it builds.
- To stop Alpha Bench, press **Ctrl+C** in the window that started it.

Your data lives in `%LOCALAPPDATA%\AlphaBench` on Windows. That folder holds:
- the alpha library
- `rules.json`
- the encrypted session
- logs

Useful launch options: `--no-browser`, `--port 8765`, `--no-keyring` (never remember credentials or sessions), and `--home <folder>`.

---

## Sign in to BRAIN

You can use every offline feature without signing in. Signing in adds:
- live simulations
- pulling your alphas and their results
- syncing the data fields and operators your account can use
- BRAIN's eligibility checks

1. Click the status pill at the top right (**Signed out**).
2. Enter your BRAIN email and password.
3. Choose whether to tick **Remember me**:
   - **Ticked:** the password goes into your OS keyring, and the session cookies are encrypted on disk. Next time you won't have to redo the face check.
   - **Unticked:** nothing is written to disk.
4. Click **Sign in**.

**If BRAIN asks for a face check**, the dialog switches to **Verification required**:

1. Click **Open verification in browser**. Or click **Copy link** and open it on your phone.
2. Complete the check. Good light, face the camera, and have your ID ready if asked.
3. Come back. Alpha Bench checks every 7 seconds (for up to 4 minutes) and signs you in when it's done. You can also click **I've completed verification**.

Alpha Bench never sees your camera, face or ID; the check is between you and BRAIN's provider. If the link expires, click **Restart sign-in** for a fresh one.

**The status pill** always shows where you stand:

| Pill | Meaning |
|---|---|
| Signed in · AB12345 | Live features available |
| Verification needed | Finish the face check |
| Session expired | Sign in again; paused jobs resume automatically |
| Sign-in paused | 3 wrong passwords; waits 15 minutes to protect your account from lockout |
| Offline | BRAIN unreachable; offline tools still work |

**Sign out and wipe local session** (in the account dialog) removes the cookies and any saved password.

Once you're signed in, do two things once from the account dialog:

- **Sync fields & operators:** Create will then only use data fields and operators your account actually has.
- **Load my active alphas:** correlation checks will then run against your real submitted alphas.

---

## The screen at a glance

| Tab | Use it to |
|---|---|
| **Create** | Generate a brand-new alpha |
| **Recreate / Modify** | Fix or rebuild an alpha you paste, or one of the Learn2Quant examples |
| **Reconcile** | Compare with BRAIN's actual result and repair what failed |
| **Library** | Browse everything you've made, with its history |
| **Correlation** | Load or paste alphas and test a candidate against them |
| **Settings & rules** | Preferences, thresholds, risk controls, course principles, calibration |

Top right:
- **BRAIN jobs** (live simulations in progress)
- the **sign-in status**
- a light/dark toggle

---

## Create an alpha

1. **Idea space:** tick the families to draw from. They are price/volume, fundamental and analyst, alternative data, event-driven, cross-sectional and time-series. Leave them all ticked for variety.
2. **Where it will trade:** choose region, universe and delay.
   - **More settings** holds decay, neutralization, truncation, pasteurization, NaN handling and the test period.
   - Defaults follow the lessons: truncation **0.01** (a 1% cap per stock) and a **1-year test period**.
   - Tick **Let Alpha Bench choose the universe** to vary it.
3. **Operator-level control** (optional): untick operators you don't want. Greyed-out ones aren't available on your account (after syncing).
4. Click **Create alpha**.

A Create run takes about 5–20 seconds:

1. Alpha Bench picks an economic thesis you haven't used recently. There are 45, including every idea from the Learn2Quant lessons.
2. It builds about a dozen candidates, simulates them offline, and screens them for correlation with your alphas and with anything created earlier in the session.
3. It repairs the best one with minimal changes and explains the result.

**Every click is different.** Alpha Bench remembers the logic of everything it has produced, and of everything in your library, and never re-derives it. It also spreads its output across the lessons' turnover buckets (low / medium / high). Earlier results stay listed under **Created this session**.

---

## Read a result

Every result shows the same things in the same order.

**1. Headline and IS summary.** PASS/FAIL, then a strip in BRAIN's own order:

> Sharpe · Turnover · Fitness · Returns · Drawdown · Margin · Returns/drawdown

Chips under the strip show the turnover bucket, and the train vs held-out test Sharpe when a test period is set.

Under the strip, a **BRAIN outlook** banner says how the offline pass is likely to land on BRAIN:

- **Good** — clears the Sharpe/fitness floors with margin.
- **Borderline** — passes offline, but inside the optimism margin (default 30%) or with a weak held-out test period. Live runs of borderline alphas often come back **"Needs Improvement"**.
- **Likely "Needs Improvement"** — your past reconciliations put the BRAIN-equivalent numbers below a floor.

The banner also lists the checks only BRAIN can run. An offline pass is never a promise; it's a screen.

**2. Expression and settings.** Everything you need to paste into BRAIN, with **Copy** buttons.

**3. Rationale:**
- why the idea should work
- how the expression is built
- why these settings
- what repairs were applied

**4. Submission checks.** One row per check: the simulated value, the **threshold that governs it**, PASS/FAIL, and (after a live run) BRAIN's own value and verdict. Open **why** on any row for the plain-language reason, suggested fixes, and where the threshold came from.

| Check | Rule (default) |
|---|---|
| Sharpe | ≥ 1.25 (delay-1), ≥ 2.0 (delay-0) |
| Fitness | ≥ 1.0 (delay-1), ≥ 1.3 (delay-0); Fitness = Sharpe × √(\|Returns\| / max(Turnover, 12.5%)) |
| Turnover | between 1% and 70% |
| Max stock weight | ≤ 10% (the weight test) |
| Sub-universe Sharpe | must hold up in the next smaller universe |
| Self-correlation | < 0.7, or Sharpe at least 10% better than the correlated alpha |
| Returns / drawdown | > 1 (Learn2Quant; advisory by default) |
| Last-2-years Sharpe | shows whether the signal is fading (advisory) |
| Production correlation | measured by BRAIN only |

**5. Other statistics:** returns, drawdown, margin, long/short counts and coverage. After you've reconciled a few alphas, a **Calibrated estimate** column also appears.

**6. Charts:** cumulative PnL, drawdown, turnover (against the 12.5% fitness floor and the 70% ceiling), rolling Sharpe (against the floor), and sector exposure. Hover to read values.

**7. Yearly table:** in BRAIN's layout (Sharpe, Turnover, Fitness, Returns, Drawdown, Margin, Long, Short).

---

## Fix a failing alpha

**Why did this fail? / How to improve** lists what fails and tests the smallest changes that could fix it, all offline. Each fix shows:
- the new Sharpe, fitness and turnover, and the change from before
- which failures it clears, and any it introduces
- an **Apply** button, which saves the fixed version as a child of the original in the library

Fixes are ordered with settings-only tweaks first (decay, neutralization, truncation, universe), then light wraps (rank, smoothing, price scaling, size-bucket neutralization), then structural ideas (such as the lessons' volatility-regime gating). Typical matches:

| Failure | Typical fixes |
|---|---|
| Fitness low, turnover high | more decay; smoothing; longer windows; `hump()`; trade only in high-volatility regimes (Lesson 4) |
| Sharpe negative | flip the sign |
| Sharpe low | finer neutralization; rank; industry-relative ranking; on BRAIN, a risk-factor neutralization (Lesson 9) |
| Weight concentrated | lower truncation; rank; backfill sparse fields |
| Sub-universe weak | neutralize within size buckets; tilt toward liquid names |
| Self-correlation high | change horizon; change neutralization; swap to a different field |
| Returns/drawdown < 1 | finer or risk neutralization; volatility scaling; lower truncation |

**Optimize settings** sweeps decay × truncation × neutralization × universe × delay. The expression is evaluated once per universe/delay, so a sweep is quick. Results are ranked with failures first, then Sharpe and fitness margins, and ◆ marks the Sharpe/fitness Pareto front. **Use** applies a row.

**Risk controls** (Settings) let you set your own turnover ceiling, position cap and sector-exposure limit. They then appear in every checks table, and Create and repair respect them.

---

## Verify on BRAIN and submit

1. On any result, click **Simulate on BRAIN** (you need to be signed in).
   - The job appears under **BRAIN jobs**. Up to 3 run at once; change this in Settings.
   - If your session expires mid-job, the job pauses and resumes after you sign in again.
2. When it finishes, the checks table gains **Verified on BRAIN** columns showing BRAIN's own numbers and verdicts. Any extra BRAIN checks appear as well.
3. Click **Check submission eligibility on BRAIN (does not submit)**. This runs BRAIN's checks, including self- and production correlation, and computes the verified correlation against your loaded active alphas.
4. When you're happy, click **Open on BRAIN to submit manually** and submit on the website.

BRAIN's actual thresholds are learned automatically, per region, from every result you pull. After that, offline checks use BRAIN's own limits.

---

## Recreate or modify an alpha

Paste an expression (multi-line expressions with variables are fine), set the settings it was simulated with, and choose:

- **Modify:** keep the alpha and apply the smallest set of changes that fixes its failures. You'll see:
  - a before/after table
  - the changes applied
  - other minimal fixes you could apply instead
- **Rebuild:** keep the *idea* but build a new alpha. Alpha Bench tries variants of your expression (horizon, normalization, field swaps, gating) and fresh builds of the matching thesis. It keeps one whose PnL correlation with the original is **below 0.7**, so it won't fail self-correlation against it.

Alpha Bench first classifies the alpha: family, thesis (e.g. "short-term reversal"), horizon, fields, and structural notes such as "no normalization".

### Learn2Quant examples

At the top of this tab, **Learn2Quant examples** lists the alphas built in WorldQuant's lessons, each with the result BRAIN showed on screen:
- **Modify** or **Rebuild** loads the alpha with its exact lesson settings.
- **Reconcile** compares Alpha Bench with the lesson's BRAIN numbers.
- **Lesson ↗** opens the video at that moment.

Several of these fail on fitness. The lessons use them as starting points, so they're good practice.

---

## Reconcile with BRAIN's real results

Use this when BRAIN disagrees with Alpha Bench, or to learn why an alpha failed.

1. **Which alpha:** pick it from the library, or enter its **BRAIN alpha ID**. When you're signed in, the expression, settings, metrics and checks are pulled automatically. Or enter the expression and settings by hand.
2. **BRAIN's results** (if not pulled by ID): paste the IS summary from BRAIN (header row and value row). Or upload a screenshot; it is read by local OCR and never uploaded (see Troubleshooting).
3. Click **Reconcile**.

You get:
- **Diagnosis:** each failure pinned to a metric. For example, "HIGH_TURNOVER: BRAIN measured 78% against a limit of 70% (Alpha Bench predicted 52%)".
- **Simulated vs verified:** every metric side by side, with the gap.
- **Why the numbers differ:** stand-in fields, turnover gaps, the delay-0 approximation, and so on.
- **Minimal fixes, predicted on BRAIN's numbers:** each fix's effect is measured offline and applied to BRAIN's real baseline. That's a shift for Sharpe, fitness and returns, and a ratio for turnover. You also see which failures each fix is predicted to clear; **Open** simulates it and saves it to the library.

Every reconciliation is logged. After three or more, results show a **Calibrated estimate** column. To seed the log immediately, use **Settings → What the course teaches → Calibrate the offline engine from the lesson results**.

---

## Check correlation with your other alphas

Self-correlation is measured on **daily PnL changes**, not cumulative PnL.

- **Load my active alphas from BRAIN** (signed in) downloads your submitted alphas and their PnL. This allows:
  - *verified* correlations for BRAIN-simulated candidates
  - *simulated* ones for offline screening
- **Paste alphas:** one per line, optionally `Name | expression`. These are simulated offline.
- **Check an expression** lists the most correlated alphas in the pool.

How to read the numbers:

| Correlation | Meaning |
|---|---|
| below 0.5 | genuinely diversifies your pool (Learn2Quant Lesson 8) |
| 0.5 to 0.7 | passes BRAIN's check but adds little diversity |
| above 0.7 | fails, unless its Sharpe beats the correlated alpha's by at least 10% |

Create automatically favours candidates below 0.5.

---

## Library and lineage

Everything you create, modify, rebuild, fix or reconcile is saved automatically: expression, settings, simulated metrics, BRAIN results and status. Click an item to see:

- **Lineage:** what it was derived from, and what was derived from it.
- **Open:** re-simulate it offline with the full result view.
- **Reconcile**, **Export JSON**, **Rename**, **Delete**.

Exports never contain credentials or cookies.

---

## Rules and thresholds

**Settings & rules → Submission rules** shows every threshold and its source.

Thresholds are resolved in this order, highest priority first:

1. **Your `rules.json`.** The path is shown on the page. Edit it and click **Reload rules file**.
2. **BRAIN check limits**, learned per region from your results. Regions differ; CHN, for example, is stricter.
3. **Built-in defaults**, labelled either with the Learn2Quant lesson they come from or as public BRAIN criteria.

If your file disagrees with what BRAIN is actually enforcing, the page warns you.

Two defaults are advisory unless you enforce them in `rules.json`:
- **returns/drawdown ≥ 1** (the course target)
- an **optional drawdown cap**, which is off by default

---

## How far to trust the offline numbers

BRAIN's data can't be downloaded, so offline results come from a **synthetic market**. It has 600 stocks with sectors, industries and sub-industries, cap-ranked universes, quarterly fundamentals, patchy coverage, and the effects the lessons exploit.

**Trust offline numbers for:**
- turnover
- weight concentration
- coverage
- the effect of settings (decay, neutralization, truncation, delay)
- ranking similar ideas against each other

**Don't trust them for** the absolute level of Sharpe and returns. Always verify on BRAIN.

How the lesson alphas compare, offline against BRAIN:

| Lesson alpha | Offline Sharpe / turnover / fitness | BRAIN (shown in lesson) |
|---|---|---|
| L4 `-group_rank(close - open, subindustry)` | 1.96 / 58% / 0.75 | 1.70 / ~58% / 0.71 |
| L9 `rank(-ts_delta(close, 2))` | 1.79 / 94% / 0.51 | 2.31 / 94% / ~0.64 |
| L5 `implied_volatility_call_720 - implied_volatility_put_720` | 1.39 / 13% / 0.93 | 1.94 / 25% / 1.44 |
| L6 `sqrt(rank(ts_mean(volume,5)/ts_mean(volume,240)))` | 2.21 / 30% / 1.10 | 1.50 / 20% / 0.84 |

Things to keep in mind:
- Truncation is scaled to BRAIN's universe size, and max weight and long/short counts are reported in BRAIN-equivalent terms.
- Drawdown follows the lessons' definition: the largest loss within any single year.
- The small synthetic universes (TOP500/TOP200) are noisy under industry neutralization. Screen on TOP3000/TOP1000.
- Fields with no offline data get a flagged stand-in.

The full list is under **Settings & rules → Simulation fidelity**.

### BRAIN says "Needs Improvement" but Alpha Bench said pass

This happens, and it has four usual causes:

1. **Pending checks.** Self-correlation against your real pool, sub-universe on real data and production correlation can only be measured by BRAIN. Offline they show as PENDING — they were never passed, just unknowable.
2. **The whole period counts.** BRAIN's verdict covers TRAIN *and* the held-out TEST year (check the TEST and IS tabs, not just TRAIN). Alpha Bench mirrors this with the held-out test Sharpe check and the enforced last-2-years Sharpe check.
3. **Optimism.** Offline Sharpe/fitness run high, which is why the outlook banner treats anything within 30% of a floor as borderline.
4. **Hidden weak years.** An aggregate Sharpe of 1.7 can hide years with fitness near 0.4; look at the yearly table.

What to do: open **Reconcile**, enter the alpha's BRAIN ID (shown under Properties on BRAIN), and reconcile. You get the exact failing checks, BRAIN's real limits (which the rules engine then adopts), fixes predicted on BRAIN's own numbers — and the comparison sharpens every future outlook. Every live simulation you run through Alpha Bench now feeds this calibration automatically.

---

## What the lessons teach, and where Alpha Bench uses it

| Lesson rule | Where it lives in Alpha Bench |
|---|---|
| Fitness = Sharpe × √(\|Returns\| / max(Turnover, 12.5%)); every lesson IS summary matches it | Fitness check and its "why" breakdown |
| Target returns/drawdown > 1 | Returns/drawdown check; Create prefers ratios above 1 |
| Drawdown = largest loss within any year of the 5-year backtest | Drawdown metric |
| Weight test and sub-universe test | Max-weight and sub-universe checks |
| Correlation low unless performance is much better | Self-correlation check (< 0.7 or +10% Sharpe) |
| Combine alphas below 0.5 correlation; diversify data, operators, holding periods | Correlation guidance; Create's diversity preference |
| Turnover buckets: low < 5%, medium 5–15%, high > 15% | Result chips; Create spreads across buckets |
| Delay-1 avoids look-ahead; delay-0 is timelier but has less capacity | Delay setting; stricter delay-0 thresholds |
| Decay smooths outliers and cuts turnover; the examples use 7–10 | Decay fixes; Create's decay choices |
| Cap any single stock at 1% | Default truncation 0.01 |
| 1-year test period in the settings dialog | Default test period; train/test Sharpe chips |
| Risk-factor neutralization: higher Sharpe, lower drawdown, slightly more turnover | Risk neutralization options; suggested fix |
| Idea types: reversion, momentum, seasonality, lead-lag, index rebalancing | Theses in Create (including industry lead-lag) |
| Data categories: price-volume, fundamental, analyst, sentiment, options, model, insider, short interest | Create's idea families and field roles |

**Settings & rules → What the course teaches** lists all of these, with the lesson each came from.

---

## Troubleshooting

| Problem | What to do |
|---|---|
| Page says "Alpha Bench is locked" | Open the exact link the launcher printed (it contains the key). |
| "Preparing the offline simulation engine" banner | First launch only; wait about 10 seconds. |
| Wrong password, then "Sign-in paused" | Wait 15 minutes; check your password on the BRAIN website first. |
| Verification link expired | Click **Restart sign-in** for a fresh link. |
| "Not verified yet" | Finish the check in your browser; Alpha Bench keeps checking, or click **I've completed verification**. |
| "BRAIN is rate-limiting requests" | Alpha Bench backs off automatically and follows BRAIN's `Retry-After`; try again shortly. |
| "Daily simulation limit reached" | Live simulations resume after BRAIN's daily reset; offline tools keep working. |
| Job shows "Paused" | Your session expired. Sign in and it resumes where it left off. |
| Screenshot upload says OCR isn't installed | Install Tesseract OCR and run `pip install -e .[ocr]`, or paste the numbers as text. |
| Create warns about placeholder field names | Sign in and **Sync fields & operators** so Create uses fields your account has. |
| Offline and BRAIN disagree a lot | Expected for absolute Sharpe; use **Reconcile**, and the calibration log will narrow the gap. |
| BRAIN says "Needs Improvement", Alpha Bench said pass | See ["BRAIN says Needs Improvement"](#brain-says-needs-improvement-but-alpha-bench-said-pass): reconcile by alpha ID; check the TEST/IS tabs; trust the outlook banner over a bare pass. |
