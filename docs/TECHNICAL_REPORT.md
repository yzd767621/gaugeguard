# GaugeGuard: technical report

OpenCV AI Competition 2026, powered by AWS. Path: Agentic Vision with OpenCV 5.

> Fill in before submitting: team name and members, repository link, live endpoint URL, video link.

## 1. Problem and users

Process plants, building services and utilities still rely on analog pressure, temperature and vacuum
gauges. They are read on walk-round inspections, written on paper, and a missed or misread value can mean an
unnoticed leak, an overheated bearing or an over-pressure event. Replacing every gauge with a transmitter is
expensive and often needs a shutdown.

A phone or fixed camera can read a gauge, but photos taken on a plant floor are dark, tilted, blurred and
full of reflections. A reader that returns a number for every photo will sometimes return a wrong number
with nothing to show it is wrong, and a wrong number that looks normal is worse than no number.

**Users.** The technician taking the photo, who needs an immediate answer or a clear instruction to retake
it. The control-room operator, who needs to trust recorded values and to approve or reject critical alerts.
The maintenance engineer, who needs an audit trail showing why each value was accepted.

**Aim.** Record a value only when the visual evidence supports it, and otherwise do the next most useful
thing: improve the image, take a second reading, ask for a new photo, or involve a person.

## 2. System overview

Three layers, shown in the diagrams in the [README](../README.md#architecture):

1. **Perception** (`gaugeguard/vision/`): OpenCV 5 image analysis.
2. **Decision** (`gaugeguard/agent/`): a planner chooses each tool call from the previous tool's output.
   Planners share one interface: a model on Amazon Bedrock (GLM-5 by default, through the Converse API;
   Claude is also supported) and a deterministic rule planner.
3. **Action** (`gaugeguard/agent/tools.py`, `store.py`): record, alert, request recapture, escalate, all
   behind guardrails written in code.

## 3. OpenCV 5 implementation

Everything below runs on `opencv-python-headless==5.0.0.93`; the Lambda build script and a unit test both
fail if the major version is not 5.

**Image quality** (`quality.py`). Laplacian variance for sharpness, mean and 5th to 95th percentile range for
brightness and contrast, Immerkaer's Laplacian-difference estimate for noise (judged relative to brightness,
so dark frames are treated fairly), and the fraction of saturated low-saturation pixels for glare. The
output is a list of named issues that the planner maps to remedies.

**Dial location** (`locate.py`). Otsu and adaptive thresholds in both polarities, `findContours`
(the TRUCO implementation in OpenCV 5), convex hull, `fitEllipse`, and a score built from the ellipse
residual, solidity and fill. A second method, `HoughCircles` with `HOUGH_GRADIENT_ALT`, tolerates a partly
hidden rim and is available to the agent as an alternative.

**Rectification** (`reader.py`). The rim is a circle seen in perspective. The rim ellipse gives an affine
correction; the needle pivot (found as the thickest ink blob near the centre with a distance transform)
then fixes the remaining projective part, because a circle together with the image of its centre determines
the plane-to-image homography up to a rotation about the centre. In the rectified frame this is a single
closed-form projective map of the disc onto itself that moves the pivot to the centre. The image is warped
once with `warpPerspective`, whose accuracy and speed were revised in OpenCV 5. A plausibility check rejects
a pivot that is inconsistent with the rim's eccentricity (for example when glare covers the hub).

**Fallbacks for real dials.** On real photos the rim found is often a bezel or case edge that is not
concentric with the scale (a dial face recessed behind its glass, seen at an angle). Three fallbacks run only
when the standard pass is unsure (confidence below 0.6):
- *Tick-line voting for the pivot.* `createLineSegmentDetector` finds line segments; tick marks are radial,
  and a projective map keeps lines concurrent, so the segments that point roughly at the centre vote for the
  point their lines pass through (refined to a quarter pixel). This finds the pivot even when the hub is a
  brass screw or hidden by glare.
- *Rectification from the tick ring.* The outer ends of the tick segments in the densest ring are mapped back
  to the photo, `fitEllipse` (with outlier rejection) gives the true dial ellipse, and rectification restarts
  from it.
- *Scale ends from tick segments.* Segments aimed at the pivot give the angular extent of the printed scale
  directly; isolated segments (screws, digits) and stray marks that make the span longer than the registry's
  nominal sweep are dropped.

If the standard and fallback passes both give a value and disagree, the better one is kept with its confidence
halved and a `reader_paths_disagree` flag.

**Scale and needle.** A black-hat style transform (morphological closing minus the image, normalised by the
local background) isolates thin dark structures independent of lighting. `warpPolar` unrolls the dial. The
tick ring is the outermost radial band with high angular high-frequency energy; the largest gap in it is the
unprinted part of the scale, which fixes the leftover rotation, so camera roll needs no assumption. The
needle is the angle with consistently high ink along a radial band inside the digits, weighted by how far ink
runs unbroken outward from the hub: real needles have long counterweight tails, and only the pointer reaches
into the tick ring, while printed digits are not attached to the hub. The opposite end of the same needle is
not counted as a rival direction; it only raises the question of which end is the pointer, which that reach
answers. A needle resting on its zero stop just below the first mark reads as the minimum. The value is
linear interpolation between the first and last tick.

**Confidence.** The product of three terms: how well the rim fits an ellipse, how closely the measured scale
sweep matches the asset's nominal sweep, and how strongly the needle peak stands out from the next best
angle. Diagnostic flags (`scale_not_found`, `scale_partially_hidden`, `pivot_inconsistent`,
`needle_outside_scale`, and others) say what went wrong.

**Enhancement** (`enhance.py`). Non-local-means denoising, percentile stretch with gamma, CLAHE on Lab
lightness, unsharp masking, and inpainting of specular highlights.

**Evidence overlay** (`annotate.py`). The rim, needle and scale ends are projected back through the inverse
homography and drawn on the photo, with the label rendered by the OpenCV 5 TrueType text engine.

## 4. Agent design

Nine tools: `get_asset_context`, `assess_quality`, `enhance_image`, `locate_dial`, `read_gauge`,
`record_reading`, `raise_alert`, `request_recapture`, `escalate_to_human`.

**How visual evidence changes the next step.** Examples from real traces in
`eval/results/example_traces.json`:

- `assess_quality` reports `dark`, so the next call is `enhance_image(denoise)` followed by a fresh
  `locate_dial` and `read_gauge`; confidence rises above the minimum and the value is recorded.
- On an occluded dial `read_gauge` returns low confidence with geometry flags, so the next call is
  `locate_dial(hough)`, and the reading from that rim is confident enough to record.
- `read_gauge` returns a confident value above the warning limit, so before recording the agent produces a
  second reading from a re-processed image. If the two agree it records and raises an alert; if not it
  escalates.
- On a motion-blurred photo, sharpening and denoising both leave confidence below the minimum, so the run
  ends with `request_recapture` and guidance to hold the camera steady and refocus.

**Guardrails in code.** They apply to both planners and are covered by tests, including a scripted model
that tries to record an unverified high value and is refused:

| Rule | Where |
|---|---|
| A reading below the minimum confidence (0.6) cannot be recorded | `record_reading` |
| An out-of-limit value, or a jump larger than the asset's `max_step`, needs an agreeing second reading from a different image version or locate method | `record_reading` |
| A reading that disagrees with another reading of the same photo in this run, or that relied on a reader fallback (section 3), needs an agreeing second reading | `record_reading` |
| A reading whose needle points at the opposite end from another reading in the run (pointer and tail confused) cannot be recorded | `record_reading` |
| A needle outside the printed scale cannot be recorded | `record_reading` |
| Alert severity is computed from the asset limits, not chosen by the planner | `raise_alert` |
| A critical alert is held until a named operator approves it | `raise_alert`, `decide_alert` |
| A recorded out-of-limit value always produces an alert, even if the planner stops early | `loop._finalise` |
| A run that ends without a decision, or exhausts its step budget, is escalated | `loop._finalise` |
| The model can only record a reading by id, never a number of its own | tool schema |

**Model planner.** `converse.py` drives any tool-capable Bedrock model through the Converse API; the
deployed default is GLM-5 (`zai.glm-5`), chosen in a live comparison of five models (section 6.2).
`bedrock.py` does the same for Claude through the Anthropic SDK, with strict tool schemas and adaptive
thinking; it is not used on the deployed account, whose AWS Free plan does not allow Anthropic models. Both
send the same system prompt and tools. The model never receives the photo, only tool results, which keeps images inside the account and
makes every decision traceable to a measurement. If the model is unreachable, throttled or declines, the
rule planner takes over from the current state and the trace records the hand-over.

**Observability.** Every run stores a trace of each tool call with its arguments, the planner's stated
reason, the result and the time taken. Runs emit CloudWatch metrics by outcome through structured logs.

## 5. AWS deployment

One arm64 (Graviton) Lambda function behind a Function URL serves the web page and the API and runs the
agent loop; Amazon Bedrock supplies the planner; DynamoDB stores runs and readings; S3 stores photos and
annotated evidence; SNS delivers alerts; CloudWatch holds logs, metrics and two alarms. The stack is defined
in `infra/template.yaml` and deploys with `infra/build_lambda.py` and `aws cloudformation package/deploy`. See [DEPLOYMENT.md](DEPLOYMENT.md).

**Status.** Deployed to ap-southeast-2 (Sydney) on 15 October 2026. Through the live URL we verified the
health check (OpenCV 5.0.0 on arm64), the bearer-token check, and a full rule-planner inspection of a
low-light image: the agent chose to denoise and recorded 67.07 against a true 67.15, about 1.5 s on the
server. The run record and evidence image were read back from DynamoDB and S3.

On 20 October 2026 the default planner was switched to GLM-5 on Bedrock and verified live on the same
image: 16 steps, about 18 s, recorded 67.10 against a true 67.15. When the model first tried to record, the
guardrail refused because the value had moved 36.5 degrees since the last recorded reading; the model then
tried the Hough locator and another rim candidate, rejected their low-confidence readings, denoised, and
recorded once two readings agreed (`eval/results/live_trace_glm5.json`). Claude could not be used: the
account is on the AWS Free plan, which does not allow Anthropic models.

## 6. Evaluation

**Data.** 660 synthetic images rendered by `gaugeguard/synth.py` with exact ground truth: five gauge designs
and 60 images for each of 11 conditions (clean, camera roll up to 35 degrees, tilt of 15 to 40 degrees,
Gaussian blur, motion blur, low light with sensor noise, noise, glare, occlusion by a bar, heavy JPEG
compression, and random combinations). Seeds below 2000 were used during development; the evaluation uses
unseen seeds from 5000. Reproduce with `python eval/run_eval.py --n 60`.

**Systems compared**, all using the same reader:

- *single pass*: locate once, read once, always report.
- *gated*: single pass, but withhold the value when confidence is below 0.6.
- *agent*: the full loop with the rule planner (deterministic, so the result is reproducible offline).

**Metric.** A reading is correct when within 2% of full scale. Percentages are of all images.

| | Value reported | Correct | Wrong value reported | Mean error of reported values | Alert recall | False alerts |
|---|---|---|---|---|---|---|
| Single pass | 96.4% | 92.9% | 3.5% | 0.58% FS | 93.6% | 5 |
| Gated | 87.9% | 87.1% | 0.8% | 0.16% FS | 86.8% | 1 |
| **Agent** | 91.1% | **90.6%** | **0.5%** | 0.13% FS | 89.1% | 1 |

What this shows:

- Always reporting gives a wrong value on 3.5% of images (23 of 660). Gating on confidence removes most of
  them but also withholds 6% of images that single pass had read correctly.
- The agent keeps wrong values at 0.5% (3 images) and reports a correct value on 90.6%. The gain over
  gating is where enhancement helps: low light goes from 82% correct (gated) to 100% (agent).
- On clean, rolled, tilted, blurred, low-light, noisy and compressed images the agent is 100% correct, with a
  mean error of 0.04% to 0.10% of full scale.
- The agent is deliberately conservative where readings of one photo disagree: glare (60% reported) and
  occlusion (67%) end in recapture requests or escalations instead of a value.
- Alert recall is lower for the agent (89.1%) than for single pass (93.6%), because the agent declines to
  alert on values it could not confirm; single pass raised 5 false alerts against the agent's 1.

Per-condition tables are in `eval/results/results.md`. Outcomes over the 660 runs: 404 recorded, 122
warnings sent, 75 critical alerts held for approval, 45 recapture requests, 14 escalations. Mean cost per
image on a laptop CPU: 7.5 tool calls and 0.39 s (95th percentile 1.05 s).

**Change from the first version.** Adding the real-photo fallbacks and the new guardrails (section 4) moved
the agent from 92.6% correct and 0.6% wrong to 90.6% correct and 0.5% wrong on this set: slightly safer,
slightly less coverage, almost all of it on glare and occlusion.

**Failure cases.** `eval/results/agent_failures.jpg` shows the images where the agent recorded a wrong value:
3 occlusion images, with errors of 2.3% to 3.6% of full scale. In the occlusion cases a bar crosses the lower part of the dial near the ends of the scale, so
the rim and scale ends are estimated from what is left; the result is slightly off but still looks
self-consistent, and the confirming second reading repeats the mistake because it comes from the same
photo. The exact mechanism has not been analysed further. Glare remains the weakest condition
for coverage (60% reported): when a reflection covers the needle no processing recovers it, and the agent
asks for a new photo.

### 6.1 Real photographs

**Data.** Two sets of openly licensed photographs (CC0, CC BY, CC BY-SA; every file credited in
`eval/real/SOURCES.md`), with values read by eye and a tolerance stated per photo:

- *Tuning set* (`eval/real/labels.json`): 14 photos from Wikimedia Commons, 9 readable and 5 that a person
  cannot read (gauge small or distant, or a symmetric needle whose pointer end is ambiguous). These photos
  were studied while improving the reader, so results on them are optimistic.
- *Held-out test set* (`eval/real/test_labels.json`): 23 photos found by a crawler (`eval/real/crawl.py`,
  Wikimedia Commons categories and Openverse, licence-filtered), filtered with the dial locator only and
  labelled before the improved reader was run on them: 21 readable (10 with a non-zero value) and 2
  ambiguous. One label was corrected after the reader had run, when its rectified view showed that the
  thick spade on a depth gauge is the counterweight (300 changed to 0); the label file says so.

For unreadable photos the only correct outcome is to withhold. Reproduce with
`python eval/run_real.py --set tuning` and `--set test`.

| Held-out test set | Readable (21): correct | Readable: wrong value | Unreadable (2): value reported |
|---|---|---|---|
| Single pass | 10 | 11 | 2 |
| **Agent** | **8** | **1** | **0** |
| Agent with the first version of the reader | 0 | 0 | 0 |

| Tuning set | Readable (9): correct | Readable: wrong value | Unreadable (5): value reported |
|---|---|---|---|
| Single pass | 5 | 4 | 4 |
| **Agent** | 1 | 0 | 0 |

What this shows:

- **The first version did not transfer to real gauges.** It found the dial but the hub check, the tick ring
  and the needle all failed on real photos, so confidence was near zero and the agent recorded nothing.
  The causes were a rim that is not concentric with the scale, dual-scale dials, and long counterweight
  tails; the fallbacks and the needle reach in section 3 address them.
- **The improved reader is usable on part of the real set.** On the held-out photos the agent recorded 9 of
  21 readable gauges: 8 within 2% of full scale (including 0.87 for a true 1.0 MPa, 5.38 for 5.8 kg/cm2 and
  9.17 for 9.7 psi) and 1 wrong (0.128 MPa for a true 0 on a 0 to 4 MPa gauge, 3.2% of full scale). It asked
  for a new photo or escalated on the other 12, and withheld on both ambiguous ones.
- **The guardrails were needed.** An earlier version of the improved reader, run before the conflict and
  pointer-end rules existed, let the agent record 4 wrong values on these photos (scored with the corrected
  label) and a value on one ambiguous photo, including 99 for a true 0 where a denoised image made the
  reader follow the counterweight. Those rules, and running the standard pass before the fallbacks, now
  refuse such readings.
- **Coverage is still low:** 43% of readable test photos get a value. Single pass is wrong on about half of
  the real photos, which is why the agent must keep withholding.

`eval/results/real_test_overlays.jpg` and `real_tuning_overlays.jpg` show each photo with the detected
rim, pivot, needle and scale ends.

### 6.2 Model planners

Five Bedrock models drove the same tools on 14 unseen synthetic gauges (one per condition plus 6 values
outside their limits), with live calls (`python eval/compare_planners.py`; full table in
`eval/results/planner_comparison.md`).

| Planner | Correct | Wrong value recorded | Needed alerts raised | Mean time per run |
|---|---|---|---|---|
| Rules | 14/14 | 0 | 6/6 | 0.3 s |
| **GLM-5** | 13/14 | 0 | 5/6 | 18.7 s |
| DeepSeek V3.2 | 13/14 | 0 | 5/6 | 19.5 s |
| gpt-oss-120b | 12/14 | 0 | 5/6 | 11.7 s |
| Kimi K2.5 | 11/14 | 0 | 3/6 | 14.8 s |
| Amazon Nova Pro | 11/14 | 0 | 4/6 | 9.5 s |

No model planner recorded a wrong value: the guardrails sit in the tools, so an unsure model can only
withhold. The models were more cautious than the rules (their misses are recapture requests or escalations)
and occasionally more accurate, because they tried more remedies. GLM-5 was chosen for accuracy, token use
and an explanation on every step. The sample is small, so differences of one or two gauges are not
significant.

## 7. Limitations

- **Limited reliability on real photographs** (section 6.1). On 21 held-out real photos the agent
  recorded a value for 9 (8 correct, 1 wrong by 3.2% of full scale) and withheld on the rest. The test set is
  small, its labels were read by eye, and most real photos available under open licences show museum
  gauges at zero. The confidence threshold was calibrated on synthetic images.
- **Small model-planner evaluation.** The 660-image numbers use the rule planner. The model planners were
  compared live on 14 gauges, one run each (section 6.2); GLM-5 takes about 15 to 20 s per run against
  0.3 s for the rules.
- **One gauge type.** A single needle on a circular dial with a tick scale and a visible gap. No dual-needle,
  linear, digital or multi-scale gauges.
- **Range comes from a registry.** The digits are not read; the asset record supplies minimum, maximum and
  nominal sweep. Selecting the wrong asset gives a wrong value with high confidence.
- **The second reading is not independent evidence.** It re-processes the same photo, so it catches
  unstable readings, not a photo that is misleading in itself (the occlusion failures above, and a real
  gauge where two image versions both followed the counterweight).
- **Linear scales only**, and the first and last ticks are assumed to be the scale's minimum and maximum.
- **The COOL path has not been attempted.**

## 8. Responsible use

- **Decision support, not a safety system.** GaugeGuard must not replace certified instrumentation, relief
  devices or interlocks. Its alerts prompt a person to look.
- **Human control.** Critical alerts require approval from a named operator; rejections are recorded.
  Uncertain cases go to a person rather than being guessed.
- **Honest uncertainty.** Values are stored with confidence and the evidence image; the system abstains
  rather than report a weak reading, and this report states its wrong-value rate.
- **Privacy.** Photos of plant equipment may show people or sensitive areas. Photos stay in a private
  encrypted bucket, expire after 30 days and are not sent to the model. Operators should frame the gauge
  only.
- **Data rights.** The synthetic evaluation images are generated by this project's own code. The 14 real
  photographs are openly licensed (CC0, CC BY, CC BY-SA) from Wikimedia Commons and are credited in
  `eval/real/SOURCES.md`; no other third-party images or datasets are used.
- **Security.** Optional bearer token on write endpoints, input size limits, reserved concurrency,
  least-privilege IAM policies scoped to the stack's own resources.

## 9. Next steps

Improve coverage on real photographs: confidence calibrated on a larger real labelled set (the crawler and
candidate filter are in `eval/real/`), reading the scale digits, and a true second photo for verification. Run the model planners on the full 660-image set and the real photos; read the scale digits to remove the
registry dependency; take a true second photo for verification when a camera can be re-triggered.
