# `md01i` — Mangdang MD01, campaign 2 (servo 6)

Copied from BAM's `bam/params/md01-3/` (working tree; the tracked name is
`bam/params/md01/`), which is the full description of the fit — read that README
before trusting these numbers.

What matters when picking one:

| | |
| --- | --- |
| Actuator | `md01i` — current setpoint `kp * 0.030 * ratio * error`, clipped to the fitted `current_limit`, `tau = kt * I` |
| Dataset | `data_md01-6v2`, servo 6 (inner coaxial shaft), 175 logs, 2026-09-21 |
| Fitted with | CMA-ES to a plateau (16k–31k trials), 12 V bench supply |
| Position MAE | 22–30 mrad held-out kp 120; 23–32 mrad over all logs |
| Working model | `m3` (also `m6`; it reproduces the measured current, `m5` does not) |

`kt` is a **gauge** in this law: rescaling it together with `error_gain_ratio` and
`current_limit` leaves the position error unchanged. The physical quantity is the
torque ceiling `kt * current_limit`, which is 0.43–0.46 Nm for m1/m3/m5/m6
(m2 0.35, m4 0.51).

Known deficiencies, from the BAM README: the two heavy blocks (≥ 0.3 Nm) fit at
31–48 mrad because the housing heated 18 °C across each kp sweep (a data confound);
`kd_position` was 0 during identification while the robot preset runs 800; and the
torque ceiling is calibrated to a 35–65 °C session.

Compared with the earlier `md01` (voltage-law) fits these are materially better on
the same data: 24–29 mrad against 50 mrad for the voltage law with a 0.45 A cap,
and 90 mrad as originally committed.

Use it as `params_file="md01i/m3"` (the directory name is the actuator name, which
is also what `resolve_params_file` expects).
