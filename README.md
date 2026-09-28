# Alpha Bench

A local tool for WorldQuant BRAIN consultants. It generates alphas, repairs failing ones, and reconciles its predictions with BRAIN's real results. It encodes BRAIN's submission rules as explicit, checkable thresholds and connects to your own BRAIN account for ground-truth simulation.

> **Your responsibility.** You're responsible for using this tool in line with your WorldQuant consultant agreement and the BRAIN terms of use. Alpha Bench only calls the BRAIN API endpoints your account can already use (sign-in, simulations, your own alphas, data fields, operators). It never scrapes the website. It never submits alphas: you submit on the BRAIN website yourself.

## Quick start (Windows)

```powershell
cd alpha-bench
.\setup.ps1          # one-time: creates a virtual environment outside OneDrive and installs dependencies
.\run.ps1            # starts Alpha Bench and opens it in your browser
```

On macOS/Linux, or without the scripts, use Python 3.11+:

```bash
python -m venv ~/.alpha-bench-venv && ~/.alpha-bench-venv/bin/pip install -e .
~/.alpha-bench-venv/bin/python -m alpha_bench
```

The launcher prints a link like `http://127.0.0.1:PORT/?k=...`. The key in that link is new on every launch, and it's what lets your browser in (see Security).

The first launch builds the offline synthetic market, which takes about 10 seconds. After that it's cached.

**New here? Read the [user guide](docs/USER_GUIDE.md).** It walks through signing in, each mode, reading results, fixing failures, verifying on BRAIN, and what the course teaches.

## Signing in to BRAIN

1. Click **Signed out** (top right). Enter your BRAIN email and password.
   - **Remember me** stores the password in the OS keyring (Windows Credential Locker, macOS Keychain or Secret Service) and the session cookies in a file encrypted with a key that also lives in the keyring.
   - Untick it and nothing is written to disk.
   - You can also set the `BRAIN_EMAIL` and `BRAIN_PASSWORD` environment variables instead.
2. **If BRAIN asks for facial verification** (Persona), Alpha Bench shows a *Verification required* screen.
   - Open the link in your own browser, or copy it to your phone, and complete the check yourself.
   - Alpha Bench checks every 7 seconds, for up to 4 minutes, whether you've finished. You can also click **I've completed verification**.
   - It never automates, captures or replays the verification.
   - If the link expires, use **Restart sign-in** to get a fresh one.
3. On the next launch, a saved session is reused if BRAIN still accepts it, so you won't redo the face check every time.
4. If the session expires mid-job, live simulations pause where they are and resume after you sign in again.
5. **Sign out and wipe local session** (in the account dialog) clears the cookies and the stored credentials.

Sign-in safeguards:
- After 3 wrong passwords, sign-in pauses for 15 minutes to protect your account from lockout. The pause survives restarts.
- On a 429 or 5xx response, Alpha Bench backs off exponentially and respects `Retry-After`.
- On `DAILY_SIMULATION_LIMIT_EXCEEDED` it stops instead of retrying.

## The three modes

| Mode | What it does |
|---|---|
| **Create** | Builds a brand-new alpha from 45 economic theses (including every idea from the Learn2Quant lessons) across price/volume, fundamental and analyst, alternative data, event-driven, cross-sectional and time-series ideas.<br><br>Each click picks a thesis you haven't used recently, with fields and structure it hasn't built before, and discards anything whose logic fingerprint was already generated or saved. It screens about a dozen candidates offline (including correlation with your alphas and with earlier creations), repairs the best with minimal changes, and explains the idea. |
| **Recreate / Modify** | Paste an expression. Alpha Bench parses it, classifies its structure (family, thesis, horizon), and then either:<br>• **modifies** it with the smallest set of edits that fixes its failures, or<br>• **rebuilds** the idea as a new alpha whose PnL correlation with the original is below 0.7. |
| **Reconcile** | Give it BRAIN's real results in one of three ways: pulled by alpha ID when signed in, pasted text, or a screenshot read by local OCR.<br><br>It diffs them against the offline prediction and pins each failure to a metric (e.g. "HIGH_TURNOVER: BRAIN measured 78% against a limit of 70% (Alpha Bench predicted 52%)"). It then proposes minimal fixes, with predicted outcomes computed on BRAIN's own numbers. Every comparison is logged to calibrate future offline estimates. |

Every result shows the expression, the exact simulation settings (region, universe, delay, decay, neutralization, truncation, pasteurization, NaN handling), and a plain-language rationale. It also shows a checks table, with each metric's threshold and the threshold's source beside it, and "Simulated" and "Verified on BRAIN" columns side by side.

Other tools:
- **Why did this fail?** gives a ranked list of minimal fixes; each can be applied in one click.
- **Optimize settings** sweeps decay, truncation, neutralization, universe and delay, and marks the Sharpe/fitness Pareto front.
- **Simulate on BRAIN** runs the alpha live on your account.
- **Check submission eligibility** runs BRAIN's checks *without* submitting.
- **Open on BRAIN** is where you submit, manually.

Everything lands in the **Library** with its lineage (what each alpha was derived from).

## Submission rules, and where the numbers come from

**The course material is built in.** The rules and worked examples come from WorldQuant's **Learn2Quant** lessons (youtube.com/@WorldQuantCareers). The transcripts were read in full, and the BRAIN screens shown in the videos (expressions, settings dialogs, IS summaries) were read from the videos' timeline thumbnails. They're recorded in `alpha_bench/course.py`. From the lessons:

- Fitness = Sharpe × √(|Returns| / max(Turnover, 12.5%)); every IS summary shown in the lessons matches it
- target returns/drawdown > 1
- drawdown is the largest loss within any year of the backtest
- the weight test and sub-universe test
- keep correlation low unless the new alpha performs much better
- combine alphas only below 0.5 correlation
- turnover buckets: low < 5%, medium 5–15%, high > 15%
- the examples cap single stocks at 1% (truncation 0.01)
- the settings dialog uses a 1-year test period
- risk-factor neutralization

The lessons don't state BRAIN's numeric submission floors, so those come from BRAIN's publicly described criteria:

- Sharpe ≥ 1.25 at delay-1 and ≥ 2.0 at delay-0
- Fitness ≥ 1.0 at delay-1 and ≥ 1.3 at delay-0
- Turnover between 1% and 70%
- No single stock above 10% of the book
- Self-correlation below 0.7, or a Sharpe at least 10% higher than the correlated alpha

Because offline numbers run optimistic, every result also carries a **BRAIN outlook** (good / borderline / likely "Needs Improvement"): a bare offline pass within 30% of a floor is flagged borderline, and once live results have been reconciled, the outlook uses your own offline-vs-BRAIN calibration instead of the fixed margin. The last-2-years Sharpe check (BRAIN's LOW_2Y_SHARPE) is enforced, and a weak held-out test period is flagged.

Every lesson alpha whose BRAIN result was shown gets the same pass/fail verdict from Alpha Bench's rules. This is covered by tests in `tests/test_course.py`.

There are three threshold sources, highest priority first:

1. **`rules.json`** in the data folder (path shown under *Settings & rules*). Put any figures you want to override here, then click **Reload rules file**.
2. **Limits BRAIN itself reports** in check results, learned automatically **per region** whenever results are pulled. Thresholds differ by region; CHN, for example, is stricter.
3. The built-in defaults, each labelled with the lesson it comes from or as public BRAIN criteria.

Settings & rules warns you when your file disagrees with what BRAIN is actually enforcing.

Two rules need care:
- The sub-universe formula (`0.75 × √(sub size / universe size) × Sharpe`) comes from community sources and is not confirmed by WorldQuant. Once an alpha is verified, BRAIN's own per-alpha limit is shown instead.
- Returns/drawdown ≥ 1 is the course's *target*, so it's advisory (WARN) until you set `"enforced": true`. An absolute drawdown cap is off unless you set one.

**Risk controls.** You can set your own turnover ceiling, position cap and sector-exposure limit under Settings. Create and repair then respect them.

## How faithful is the offline engine?

BRAIN's data can't be downloaded, so offline screening runs on a **synthetic market**:
- 600 stocks with sector, industry and sub-industry groups
- cap-ranked universes with monthly churn
- fundamentals that update quarterly, point-in-time
- patchy analyst and news coverage
- a dozen weak, regime-varying anomalies (tuned so the Learn2Quant lesson alphas behave as they did on BRAIN)

**What transfers well to BRAIN:**
- turnover
- weight concentration and coverage
- the effect of decay, truncation, neutralization and delay
- the relative ranking of similar ideas

**What does not transfer:** absolute Sharpe and returns.

The mechanics mirror BRAIN:
- expression → pasteurize → linear decay → neutralize → scale to a $20M book → iterative truncation
- delay-d shifts every input field
- Returns = annual PnL / $10M
- Fitness = Sharpe × √(|Returns| / max(Turnover, 12.5%))
- self-correlation is measured on *daily* PnL changes

Fields that aren't in the synthetic market (e.g. a real `fnd6_...` id) get deterministic stand-ins shaped by their dataset category, and the result is flagged. The full list of approximations is under *Settings & rules → Simulation fidelity*. Reconciliation shows how far off the offline numbers were for your alphas, and a calibration log narrows the gap over time.

## Security

- The server binds to `127.0.0.1` only.
- A random per-launch key arrives once via the launch link and is then kept in an HttpOnly, SameSite=Strict cookie.
- Every API call also requires a custom header, so other websites can't drive the API.
- The Host header is checked to block DNS-rebinding attacks.
- A strict Content-Security-Policy is set, and the page uses no third-party scripts, fonts or CDNs.
- Passwords and cookies never appear in logs, error messages or exports. A redaction filter masks them even in debug output.
- Validation errors never echo request bodies back.
- BRAIN is only ever called over HTTPS with certificate verification on.
- Nothing — credentials, cookies, alpha expressions or results — is sent to any third party, and there's no telemetry.
- Screenshots for Reconcile are read with local Tesseract OCR only (`pip install -e .[ocr]` plus the Tesseract program). Without it, paste the numbers as text.

## Development

```powershell
& "$env:LOCALAPPDATA\AlphaBench\venv\Scripts\python.exe" -m pytest
```

The suite has 122 tests. BRAIN is faked in all of them (`tests/fakes.py`: success, inquiry-required, Location-header verification, 401, 429, 5xx), so they never touch the network, your keyring or your simulation quota.

| Folder | Contents |
|---|---|
| `alpha_bench/brain/` | Sign-in state machine, keyring and encrypted session store, API client, job queue (concurrency cap, default 3) |
| `alpha_bench/expr/` | Fast Expression parser and printer, operator catalog, fingerprints, structure classifier |
| `alpha_bench/engine/` | Synthetic market, evaluator (95 operators), portfolio simulator and metrics |
| `alpha_bench/rules/` | Rules engine and correlation |
| `alpha_bench/gen/` | Idea library, generator, repair, recreate, sweep, reconcile |
| `alpha_bench/store/` | SQLite library (versions, lineage, jobs, reconciliation log) |
| `alpha_bench/web/` | Local server and UI |

**API assumptions.** Endpoint shapes were checked against WorldQuant's ACE helper library and several open-source BRAIN clients:
- both Persona challenge shapes
- the `regular.code` object
- `Accept: application/json;version=2.0` for correlation and check endpoints
- the terminal simulation statuses

If BRAIN's live responses differ, the places to adjust are `parse_challenge` in `brain/auth.py` and `brain/client.py`.
