# Hispanic Heritage Historical Data Explorer

A self-contained Streamlit app for authentic World Bank historical exploration and a changeable, course-aware mathematical function lab. No credentials, missing data files, or custom local modules are required by app.py.

The prompt left the research question, countries and pathway unfilled. Defaults are an editable question about Mexico's life expectancy, Mexico, life expectancy, and Calculus. Choose your assigned pathway in the sidebar and edit the question/selections before presenting. Brazil is offered as a Portuguese-speaking Latin American comparison, not a Spanish-speaking Hispanic country.

## Files and local start

- `app.py`: complete application; all application logic is here.
- `requirements.txt`: minimal, unpinned direct dependencies for deployment.
- `requirements`: identical extensionless copy requested by the team. Keep `requirements.txt` in GitHub: hosts use that conventional name.
- `sources.md`: official data mappings, definitions, documentation and provenance rules.
- `test_app.py`, `test_ui.py`: optional reproducible verification; not imported by the app.
- `verification.md`: recorded tests, environment, and remaining classroom/deployment checks.

Use Python 3.12 (the tested version), ideally a virtual environment. Open a terminal in the extracted folder:

```bash
python -m venv .venv
```

Activate it on macOS/Linux with `source .venv/bin/activate`, or Windows PowerShell with `.venv\Scripts\Activate.ps1`. Then:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in the terminal. If `pip` points to another Python, use `python -m pip install -r requirements.txt` instead.

## GitHub and Streamlit Community Cloud

1. Create a GitHub repository, for example `hispanic-heritage-data-explorer`.
2. Upload `app.py`, `requirements.txt`, `README.md` and `sources.md` to the repository root and commit. Tests are optional; do not upload `.venv`, caches or credentials.
3. In Streamlit Community Cloud, connect GitHub, choose the repository and branch, and set the main file path to `app.py`.
4. Select Python 3.12 in advanced settings when offered, deploy, and inspect build/app logs.
5. Open the deployed URL and run the verification checklist below. No API key or secrets are needed. Public World Bank access from the hosting environment is required.

Official guides: [Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [dependency files](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Render

Create a **Python Web Service**, connect the GitHub repository, select its branch, and leave the root directory empty when the files are in the repository root. Set Python 3.12 in Render's environment configuration using the supported full patch version for your service. Read the service's build and runtime logs.

Build command:

```bash
pip install -r requirements.txt
```

Start command, on ONE LINE:

```bash
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

Equivalent if the Streamlit executable is on PATH:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

Use a Web Service because this app runs Python and maintains interactive sessions. Render requires a listener on `0.0.0.0` and recommends its `PORT` variable. See [Render Web Services](https://render.com/docs/web-services). The repository has not been deployed to either hosting service as part of this deliverable.

## First classroom walkthrough

1. Choose your course, country and indicator. Read the definition and units before interpreting the chart.
2. Check the requested 70-calendar-year window against actual coverage/count. Expand the original table and metadata. Open the exact API URL to inspect observations.
3. Edit a working value. Observe the user-modified scenario label. Reset to restore the source. Incomplete, duplicate, noninteger, nonnumeric or indicator-invalid rows block modeling.
4. Change degree, sampling increment, horizon and comparison countries. Degree changes the fitted model; sampling changes markers/table; horizon changes the forward segment; comparisons show source series and coverage gaps, not independently fitted forecasts.
5. Calculate one year, then a batch including the same year. The estimates must agree. Saved calculations survive ordinary reruns and clear after country, indicator, degree or working-data changes.
6. Open the Function Lab. Select a context and use Suggested function, Parameter builder, Custom expression or Historical best fit. Historical mode uses centered years, **not** calendar years as x.
7. Set the interval, point, secant endpoints and graph toggles. Play, pause, reset and change speed. Point, tangent, numerical values, explanations and schematic share the same current x.
8. Download CSV, Excel and PDF from the historical report. Check data status, units, fitted range and calculated rows. Function Lab teaching models are deliberately not exported as World Bank observations.

## Safe function notation

Use `x`, finite constants, `pi`, `e`, parentheses, `+ - * /`, `^` or `**`, and `sin`, `cos`, `tan`, `exp`, `ln`/`log`, `sqrt`. Trigonometric arguments are in radians. Always write multiplication: `2*x`, not `2x`. Natural logarithms use `ln(x)` or `log(x)`.

Examples:

| Family | Expression | Important restriction |
| --- | --- | --- |
| Polynomial | `x^3 - 3*x` | All real x |
| Rational | `1/(x-2)` | x cannot be 2 |
| Exponential | `100*exp(0.2*x)` | All real x; overflow possible for large x |
| Logarithmic | `ln(x)` | x > 0 |
| Trigonometric | `20+5*sin(pi*x/12)` | All real x |
| Tangent | `tan(x)` | x cannot be pi/2 + k*pi |
| Root | `sqrt(x)` | x >= 0; finite classical derivative unavailable at 0 |

Expressions are at most 180 characters and 90 syntax nodes. Powers are constant integer/half-integer exponents between -8 and 8. Use `exp(k*x)` for exponential functions. No Python statements, attributes, imports, arbitrary calls, arbitrary variable exponents, piecewise definitions or implicit multiplication are accepted. A restricted AST walker constructs SymPy objects; it never evaluates user-entered Python. This is an educational input boundary, not a sandbox for executing arbitrary code.

## Important code and data flow

### Source and validation

`fetch_data` calls the official World Bank V2 country/indicator endpoint using `date=current_year-70:current_year-1`, JSON and a page size larger than the requested window. It checks HTTP errors, response shape and pagination and retains null values. It caches successful responses for six hours, with retrieval time and actual URL. `fetch_metadata` separately caches official definitions for 24 hours. Refresh source clears both caches. An error is displayed; no random data are substituted. Missing observations are omitted only from the fitting table, without interpolation or imputation.

`validate_table` creates a sorted copy of the editable year/value table. Years must be unique integers in 1950-2100 and both columns finite. Rates/counts that must be nonnegative are constrained accordingly; unemployment is 0-100%; inflation permits deflation down to -100%; net migration and external balance can be negative. The life-expectancy ceiling of 150 and birth-rate ceiling of 1,000 are broad classroom plausibility guards, not official World Bank validation rules. GDP/debt/reserves have no arbitrary upper cap. Source observations stay in a separate table, and Reset uses that source.

### The historical polynomial

`fit_model` sets `base = mean(observed years)` and `t = year - base`. It uses NumPy's scaled polynomial least-squares fitter and converts once to coefficients in ordinary t powers. The stored Polynomial then drives `predict` for every graph, single-year result, batch, sample and export. The equation shows 16 significant digits; do not rebuild the model from rounded display text.

At least degree+1 distinct years are required. A degree-d polynomial has up to d-1 turning points, but may have fewer. High degree can capture noise, and extrapolation can explode. Least squares minimizes squared residuals and assumes only a numerical approximation for descriptive purposes; no independence, normality or constant variance is asserted for these time series.

Residual = observation - fit. MAE is mean absolute residual, in the indicator's units. R-squared compares squared residuals with variation around the observed mean. In-sample metrics measure agreement with training observations, not generalization. R-squared can be negative on holdout; constant observed series make its interpretation uninformative. With sufficient data, the final max(3, ceil(20% of n)) observations form a chronological test set, and a separate polynomial is trained on earlier years. Holdout MAE/R-squared are labeled separately; the displayed final model still refits all working data. No random split mixes future values into training.

`results_table` labels within/outside fitted range, distinguishes fitted values at observed years from actual working observations, and flags violated indicator bounds without clipping. A value at an observed year is still a prediction; its error need not be zero. `parse_years` reports malformed entries and retains valid requested integers. A hash of source, selection, degree and working table invalidates stale saved results. Session state persists within the session, not after closing the session/server restart.

### Domain and calculus

`parse_function` preserves unevaluated operations so restrictions like `x/x` at zero are not erased by initial algebraic cancellation. `domain_for` asks SymPy for the real continuous domain. `domain_segments` intersects that domain with the selected finite interval and explicitly resolves periodic exclusions such as tangent poles. Each segment is plotted separately, using NaNs to break unresolved numerical jumps. An unresolved symbolic domain is labeled as such; no completeness claim is made.

`analysis_points` computes symbolic derivatives, then uses NumPy polynomial roots or SciPy's Brent root finder on sign brackets for general functions. A sign change + to - in f' indicates a local maximum; - to + indicates a local minimum. A stationary point without a sign change is not labeled an extremum. Inflection candidates require finite f and a numerical sign change in f''. `checked_slope` adds a one-sided numerical consistency guard so obvious corners do not get an invented tangent. Domain endpoints of derivatives can add nondifferentiable critical candidates when SymPy resolves them.

All displayed root locations and classifications are **numerical approximations**, not exact solutions. Polynomial roots are sought including repeated roots; roundoff may affect multiplicity. Sign scanning for nonpolynomial functions may miss touching roots, rapid oscillations or very close roots. Sign intervals are midpoint tests between detected candidates and are not a proof that all roots were found. At a corner, symbolic distribution terms (such as DiracDelta) do not establish ordinary second derivatives. Use a simpler interval/function or confirm such cases by hand.

For a function established continuous on the closed interval, absolute-extrema candidates include endpoints plus detected critical points. For general functions these remain candidates unless completeness is independently established. The app never claims a universal symbolic solver for every expression. Unsupported or unresolved cases are labeled, rather than silently assigned invented answers.

### Course levels and interpretation

- Honors Algebra 2: domain, observed/sampled range, polynomial end behavior, graphical trends and average rates/secants.
- Precalculus: also degree/coefficients and changes in average rates as numerical evidence about turning and concavity.
- Calculus: symbolic first/second derivatives, tangent and secant, critical classifications, increasing/decreasing intervals, concavity, verified numerical inflection candidates, and endpoint-aware extrema candidates.

If f has units U and x has units T, f' has U/T and f'' has U/T². For position in meters and time in seconds, these are velocity (m/s) and acceleration (m/s²). For economic output per year, the first derivative is annual change; the second describes how that annual change changes. These describe the chosen model, not historical causes.

### Synchronization and visuals

`lab_ui` uses a Streamlit fragment with a 0.3-second rerun interval. During playback it advances one shared x by a selected percentage of the interval per tick and wraps after the right endpoint. Play/Pause/Reset update that same state. The graph point, tangent, values, contextual interpretation and `simulation` all consume it. Browser scheduling affects actual frame rate; speed is model advancement per tick, not a guaranteed physical clock.

The six visuals are stylized SVG schematics, including a person with a prosthetic leg. They are not biomechanical or pharmacological simulations. Position, fill or cell count responds to f(x), scaled by the 5th-95th percentiles of finite sampled values on the interval. This visual scale saturates outside that range; numbers and model predictions are never clipped. Changing function or context therefore changes the simulation meaningfully while keeping the scaling explicit.

### Exports

`make_chart` plots observations, smooth historical fit, historical boundaries, separate extrapolation, sampled markers and saved estimates. Comparison source nulls create gaps. `export_excel` writes real zipped Office XML with XlsxWriter: Summary, Working data, Original source, Calculated years, Model samples and Comparisons sheets. `export_pdf` creates a real ReportLab PDF using Matplotlib's bundled font, report metadata, equation, metrics, chart, limitations, preview and all calculated results. CSV includes labeled observation/calculation rows and model/provenance columns. Excel formula interpretation is disabled for entered text.

## Verification and diagnosis

Run reproducible tests from this folder:

```bash
python -m unittest test_app.py test_ui.py
```

See `verification.md` for what was actually executed. Unit-test fixtures are explicitly artificial and are never supplied by the running app as source data.

| Symptom | Check and response |
| --- | --- |
| ModuleNotFoundError | Confirm the right virtual environment and `python -m pip install -r requirements.txt`; inspect the first missing package in logs. |
| `streamlit` command not found | Use `python -m streamlit run app.py`. |
| Deployment cannot find app.py | Check case, repository root, selected branch and entrypoint. |
| API timeout, HTTP error or no data | Read the source error, open the exact URL, try Refresh or another indicator, and check host outbound HTTPS. Do not invent replacements. |
| Too few observations | Lower degree or select a series with better coverage. |
| Invalid table | Remove incomplete/duplicate rows or correct numeric constraints; Reset restores source. |
| Custom function rejected | Use explicit *, allowed functions and bounded powers; remove unsupported Python notation. |
| Missing graph around a pole | This is intentional. Reduce the interval and inspect the displayed domain. |
| Overflow/undefined estimate | Reduce the interval/horizon or revise parameters. Undefined numbers are not silently clipped. |
| Slow or uneven animation | Network/browser scheduling and plot rendering affect tick rate; pause when discussing results. |
| Render port detection failure | Keep the complete one-line start command with `0.0.0.0` and `$PORT`; inspect runtime logs. |
| PDF or workbook differs from expectations | Recalculate after edits; confirm scenario status and distinguish saved calculations from unsaved inputs. |

Unpinned requirements follow the requested Streamlit-friendly format. The tested versions in `verification.md` document one compatible environment, not a promise that future versions will remain compatible. After dependency updates, rerun the tests and deployed checklist.
