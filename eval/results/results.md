# Evaluation results

OpenCV 5.0.0, 60 synthetic images per condition (660 total), seeds from 5000. A reading is correct when it is within 2.0% of full scale. All percentages are of all images in the group.

## Overall

| | Reported % | Correct % | Wrong value reported % | MAE of reported (% FS) | Alert recall % | False alerts |
|---|---|---|---|---|---|---|
| single_pass | 96.4 | 92.9 | 3.5 | 0.58 | 93.6 | 5 |
| gated | 87.9 | 87.1 | 0.8 | 0.16 | 86.8 | 1 |
| agent | 91.1 | 90.6 | 0.5 | 0.13 | 89.1 | 1 |

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
| single_pass | 100.0 | 98.3 | 1.7 | 0.74 | 100.0 | 1 |
| gated | 85.0 | 85.0 | 0.0 | 0.56 | 85.0 | 1 |
| agent | 86.7 | 86.7 | 0.0 | 0.5 | 85.0 | 1 |

### low_light

| | Reported % | Correct % | Wrong value reported % | MAE of reported (% FS) | Alert recall % | False alerts |
|---|---|---|---|---|---|---|
| single_pass | 100.0 | 98.3 | 1.7 | 0.21 | 100.0 | 1 |
| gated | 81.7 | 81.7 | 0.0 | 0.1 | 90.0 | 0 |
| agent | 100.0 | 100.0 | 0.0 | 0.06 | 100.0 | 0 |

### noise

| | Reported % | Correct % | Wrong value reported % | MAE of reported (% FS) | Alert recall % | False alerts |
|---|---|---|---|---|---|---|
| single_pass | 100.0 | 100.0 | 0.0 | 0.08 | 100.0 | 0 |
| gated | 98.3 | 98.3 | 0.0 | 0.08 | 100.0 | 0 |
| agent | 100.0 | 100.0 | 0.0 | 0.07 | 100.0 | 0 |

### glare

| | Reported % | Correct % | Wrong value reported % | MAE of reported (% FS) | Alert recall % | False alerts |
|---|---|---|---|---|---|---|
| single_pass | 81.7 | 68.3 | 13.3 | 2.0 | 70.0 | 1 |
| gated | 53.3 | 50.0 | 3.3 | 0.41 | 55.0 | 0 |
| agent | 60.0 | 60.0 | 0.0 | 0.15 | 65.0 | 0 |

### occlusion

| | Reported % | Correct % | Wrong value reported % | MAE of reported (% FS) | Alert recall % | False alerts |
|---|---|---|---|---|---|---|
| single_pass | 78.3 | 60.0 | 18.3 | 3.51 | 75.0 | 2 |
| gated | 60.0 | 55.0 | 5.0 | 0.53 | 60.0 | 0 |
| agent | 66.7 | 61.7 | 5.0 | 0.44 | 65.0 | 0 |

### jpeg

| | Reported % | Correct % | Wrong value reported % | MAE of reported (% FS) | Alert recall % | False alerts |
|---|---|---|---|---|---|---|
| single_pass | 100.0 | 100.0 | 0.0 | 0.05 | 95.0 | 0 |
| gated | 100.0 | 100.0 | 0.0 | 0.05 | 95.0 | 0 |
| agent | 100.0 | 100.0 | 0.0 | 0.05 | 95.0 | 0 |

### combined

| | Reported % | Correct % | Wrong value reported % | MAE of reported (% FS) | Alert recall % | False alerts |
|---|---|---|---|---|---|---|
| single_pass | 100.0 | 96.7 | 3.3 | 0.51 | 100.0 | 0 |
| gated | 88.3 | 88.3 | 0.0 | 0.15 | 80.0 | 0 |
| agent | 88.3 | 88.3 | 0.0 | 0.11 | 80.0 | 0 |

## Agent outcomes

- alert_pending_approval: 75
- alert_raised: 122
- escalated: 14
- recapture_requested: 45
- recorded: 404

Agent cost on this machine: mean 7.45 tool calls and 393.9 ms per image (p95 1054.1 ms).
