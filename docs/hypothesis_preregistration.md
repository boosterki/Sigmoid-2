# Pre-registration — Workstream 2

Written by `bikeops.inference.preregister()` on 2026-11-04 07:47, before any model in
Workstream 2 was estimated.

**H1.** Holding hour, calendar class, month and trend fixed, wet weather (weathersit 3, with 4 folded in)
reduces casual rentals by a substantially larger proportion than registered rentals.

**H0.** The wet-weather rate-ratio is the same for both segments.

**Test statistic.** Ratio of rate-ratios, RRR = RR(casual, wet vs dry) / RR(registered, wet vs dry).

**Decision rule.** Reject H0 in favour of H1 only if BOTH
1. the Negative Binomial GLM (date-clustered standard errors) gives RRR < 0.75 with a 95% CI excluding 1, and
2. the matched wet-vs-dry comparison gives RRR < 0.75 with a 95% bootstrap CI (over days) excluding 1.

If only one method meets the rule, the result is reported as *inconclusive*; if neither, H0 is not rejected.
Robustness checks (each year separately, excluding disruption days, mist as a second treatment) are
reported but do not change the decision.
