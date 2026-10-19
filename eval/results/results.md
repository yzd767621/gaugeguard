# Evaluation results

OpenCV 5.0.0, 60 synthetic images per condition (660 total), seeds from 5000. A reading is correct when it is within 2.0% of full scale. All percentages are of all images in the group.

## Overall

| | Reported % | Correct % | Wrong value reported % | MAE of reported (% FS) | Alert recall % | False alerts |
|---|---|---|---|---|---|---|
| single_pass | 96.5 | 91.5 | 5.0 | 0.7 | 93.6 | 4 |
| gated | 81.1 | 80.5 | 0.6 | 0.14 | 80.5 | 1 |
| agent | 93.2 | 92.6 | 0.6 | 0.14 | 89.5 | 1 |

## By condition

### clean

| | Reported % | Correct % | Wrong value reported % | MAE of reported (% FS) | Alert recall % | False alerts |
|---|---|---|---|---|---|---|
| single_pass | 100.0 | 100.0 | 0.0 | 0.05 | 100.0 | 0 |
| gated | 100.0 | 100.0 | 0.0 | 0.05 | 100.0 | 0 |
| agent | 100.0 | 100.0 | 0.0 | 0.05 | 100.0 | 0 |

### roll

| | Reported % | Correct % | Wrong value reported % | MAE of reported (% FS) | Alert recall % | False alerts |
|---|---|---|---|---|---|---|
| single_pass | 100.0 | 100.0 | 0.0 | 0.04 | 95.0 | 0 |
| gated | 100.0 | 100.0 | 0.0 | 0.04 | 95.0 | 0 |
| agent | 100.0 | 100.0 | 0.0 | 0.04 | 95.0 | 0 |

### tilt

| | Reported % | Correct % | Wrong value reported % | MAE of reported (% FS) | Alert recall % | False alerts |
|---|---|---|---|---|---|---|
| single_pass | 100.0 | 100.0 | 0.0 | 0.05 | 100.0 | 0 |
| gated | 100.0 | 100.0 | 0.0 | 0.05 | 100.0 | 0 |
| agent | 100.0 | 100.0 | 0.0 | 0.05 | 100.0 | 0 |

### blur

| | Reported % | Correct % | Wrong value reported % | MAE of reported (% FS) | Alert recall % | False alerts |
|---|---|---|---|---|---|---|
| single_pass | 100.0 | 100.0 | 0.0 | 0.1 | 95.0 | 0 |
| gated | 100.0 | 100.0 | 0.0 | 0.1 | 95.0 | 0 |
| agent | 100.0 | 100.0 | 0.0 | 0.1 | 95.0 | 0 |

### motion

| | Reported % | Correct % | Wrong value reported % | MAE of reported (% FS) | Alert recall % | False alerts |
|---|---|---|---|---|---|---|
| single_pass | 100.0 | 95.0 | 5.0 | 0.78 | 100.0 | 1 |
| gated | 81.7 | 81.7 | 0.0 | 0.52 | 85.0 | 1 |
| agent | 88.3 | 88.3 | 0.0 | 0.52 | 85.0 | 1 |

### low_light

| | Reported % | Correct % | Wrong value reported % | MAE of reported (% FS) | Alert recall % | False alerts |
|---|---|---|---|---|---|---|
| single_pass | 100.0 | 95.0 | 5.0 | 0.38 | 100.0 | 0 |
| gated | 45.0 | 45.0 | 0.0 | 0.06 | 60.0 | 0 |
| agent | 100.0 | 100.0 | 0.0 | 0.05 | 100.0 | 0 |

### noise

| | Reported % | Correct % | Wrong value reported % | MAE of reported (% FS) | Alert recall % | False alerts |
|---|---|---|---|---|---|---|
| single_pass | 100.0 | 100.0 | 0.0 | 0.06 | 100.0 | 0 |
| gated | 86.7 | 86.7 | 0.0 | 0.06 | 85.0 | 0 |
| agent | 100.0 | 100.0 | 0.0 | 0.05 | 100.0 | 0 |

### glare

| | Reported % | Correct % | Wrong value reported % | MAE of reported (% FS) | Alert recall % | False alerts |
|---|---|---|---|---|---|---|
| single_pass | 83.3 | 71.7 | 11.7 | 1.9 | 80.0 | 1 |
| gated | 48.3 | 48.3 | 0.0 | 0.17 | 50.0 | 0 |
| agent | 70.0 | 70.0 | 0.0 | 0.14 | 65.0 | 0 |

### occlusion

| | Reported % | Correct % | Wrong value reported % | MAE of reported (% FS) | Alert recall % | False alerts |
|---|---|---|---|---|---|---|
| single_pass | 78.3 | 51.7 | 26.7 | 4.73 | 75.0 | 2 |
| gated | 53.3 | 48.3 | 5.0 | 0.54 | 55.0 | 0 |
| agent | 78.3 | 73.3 | 5.0 | 0.42 | 70.0 | 0 |

### jpeg

| | Reported % | Correct % | Wrong value reported % | MAE of reported (% FS) | Alert recall % | False alerts |
|---|---|---|---|---|---|---|
| single_pass | 100.0 | 100.0 | 0.0 | 0.05 | 95.0 | 0 |
| gated | 100.0 | 100.0 | 0.0 | 0.05 | 95.0 | 0 |
| agent | 100.0 | 100.0 | 0.0 | 0.05 | 95.0 | 0 |

### combined

| | Reported % | Correct % | Wrong value reported % | MAE of reported (% FS) | Alert recall % | False alerts |
|---|---|---|---|---|---|---|
| single_pass | 100.0 | 93.3 | 6.7 | 0.62 | 90.0 | 0 |
| gated | 76.7 | 75.0 | 1.7 | 0.17 | 65.0 | 0 |
| agent | 88.3 | 86.7 | 1.7 | 0.15 | 80.0 | 0 |

## Agent outcomes

- alert_pending_approval: 76
- alert_raised: 122
- escalated: 2
- recapture_requested: 43
- recorded: 417

Agent cost on this machine: mean 7.11 tool calls and 298.7 ms per image (p95 767.7 ms).
