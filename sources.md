# Sources and interpretation

Documentation checked and live default API tested on 2026-10-15 UTC. Each running app request records its own actual retrieval timestamp; the date in this file is not a substitute for that provenance.

## Official data

All historical observations come from the [World Bank Indicators API V2](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392). The app uses the [documented JSON country/indicator structure](https://datahelpdesk.worldbank.org/knowledgebase/articles/898581-api-basic-call-structures) and [indicator metadata endpoint](https://datahelpdesk.worldbank.org/knowledgebase/articles/898599-indicator-api-queries). Underlying producing organizations differ by indicator; the app displays `sourceOrganization`, `sourceNote` and the World Bank's last-update date when available.

Country codes: Mexico/MX, Brazil/BR, Argentina/AR, Colombia/CO, Chile/CL, Peru/PE, Venezuela/VE, Ecuador/EC, Uruguay/UY. Brazil provides a Latin American comparison and is Portuguese-speaking.

| Indicator and official link | Code | Units | Meaning / caution |
| --- | --- | --- | --- |
| [Life expectancy at birth, total](https://data.worldbank.org/indicator/SP.DYN.LE00.IN) | SP.DYN.LE00.IN | years | Years a newborn would live under prevailing age-specific mortality patterns. |
| [Unemployment, total, national estimate](https://data.worldbank.org/indicator/SL.UEM.TOTL.NE.ZS) | SL.UEM.TOTL.NE.ZS | % of labor force | National estimate; do not substitute modeled ILO series or assume identical national methods. |
| [GDP per capita](https://data.worldbank.org/indicator/NY.GDP.PCAP.CD) | NY.GDP.PCAP.CD | current US$ per person | Output per person, not household income or average wealth; nominal dollars are not inflation-adjusted. |
| [Crude birth rate](https://data.worldbank.org/indicator/SP.DYN.CBRT.IN) | SP.DYN.CBRT.IN | births per 1,000 people | Rate, not birth count. |
| [Net migration](https://data.worldbank.org/indicator/SM.POP.NETM) | SM.POP.NETM | people | Immigrants minus emigrants, not total emigrants; use the current official methodology note for period/estimation details. |
| [Intentional homicides](https://data.worldbank.org/indicator/VC.IHR.PSRC.P5) | VC.IHR.PSRC.P5 | homicides per 100,000 people | Rate, not total homicide count. |
| [Consumer price inflation](https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG) | FP.CPI.TOTL.ZG | annual % | CPI annual percentage change; deflation can be negative. |
| [Central government debt](https://data.worldbank.org/indicator/GC.DOD.TOTL.GD.ZS) | GC.DOD.TOTL.GD.ZS | % of GDP | Central government coverage, not necessarily all public debt; gaps are common. |
| [Total reserves including gold](https://data.worldbank.org/indicator/FI.RES.TOTL.CD) | FI.RES.TOTL.CD | current US$ | Reserve assets including monetary gold; not government income. |
| [External balance on goods and services](https://data.worldbank.org/indicator/NE.RSB.GNFS.ZS) | NE.RSB.GNFS.ZS | % of GDP | Exports minus imports; not the current-account balance. |

Example default request in 2026:

https://api.worldbank.org/v2/country/MX/indicator/SP.DYN.LE00.IN?date=1956%3A2025&format=json&per_page=1000

The app recomputes the window as UTC current year minus 70 through current year minus 1 inclusive. A 70-year request is not a promise of 70 observations. Nulls are retained in original records, not filled; missing years are disclosed. Caches may retain successful responses for six hours until refresh. Live metadata may be cached for 24 hours. Values can be revised later.

Edited data are user-modified scenarios, never re-labeled as World Bank observations. Suggested context functions and parameter models are classroom-created examples with no claim of empirical calibration. No schooling series, current-account series or historical causal explanation is fabricated.

## Mathematical and software references

- [NumPy Polynomial.fit](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.fit.html): scaled least-squares fitting; conversion gives coefficients in the chosen ordinary variable.
- [scikit-learn regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics): MAE and R-squared.
- [SciPy brentq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html): bracketed scalar root finding. SciPy is a direct dependency specifically for this numerical analysis.
- [SymPy calculus](https://docs.sympy.org/latest/modules/calculus/index.html): symbolic derivatives and continuous-domain analysis. Symbolic domain resolution and numerical root finding have limitations documented in the app.
- [Streamlit fragments](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment): timed fragment reruns for synchronized playback.
- [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing): programmatic widget/runtime verification.
- [Plotly Python](https://plotly.com/python/): interactive function charts.
- [Matplotlib](https://matplotlib.org/stable/): static historical charts and bundled DejaVu font.
- [XlsxWriter with pandas](https://xlsxwriter.readthedocs.io/working_with_pandas.html): genuine Excel workbook generation.
- [ReportLab documentation](https://docs.reportlab.com/): PDF document/chart/table generation.
- [Streamlit deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency file convention](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).
- [Render Python Web Services and port binding](https://render.com/docs/web-services).

No inferential prediction intervals are implemented. Residuals and MAE describe errors; they do not automatically yield 95% confidence or prediction intervals for time-dependent, potentially nonstationary historical series.
