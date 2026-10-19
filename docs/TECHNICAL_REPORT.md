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

**Scale and needle.** A black-hat style transform (morphological closing minus the image, normalised by the
local background) isolates thin dark structures independent of lighting. `warpPolar` unrolls the dial. The
tick ring is the outermost radial band with high angular high-frequency energy; the largest gap in it is the
unprinted part of the scale, which fixes the leftover rotation, so camera roll needs no assumption. The
needle is the angle with consistently high ink along a radial band inside the digits. The value is linear
interpolation between the first and last tick.

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
| Single pass | 96.5% | 91.5% | 5.0% | 0.70% FS | 93.6% | 4 |
| Gated | 81.1% | 80.5% | 0.6% | 0.14% FS | 80.5% | 1 |
| **Agent** | 93.2% | **92.6%** | **0.6%** | 0.14% FS | 89.5% | 1 |

What this shows:

- Always reporting gives a wrong value on 5.0% of images (33 of 660). Gating on confidence removes almost
  all of them but also withholds 11% of images that single pass had read correctly.
- The agent recovers most of that loss: it reports a correct value on 92.6% of images while keeping wrong
  values at 0.6% (4 images). The gain is concentrated where enhancement helps: low light goes from 45% to
  100% correct relative to gating, noise from 87% to 100%, combined degradations from 75% to 87%.
- On clean, rolled, tilted, blurred and compressed images all three systems are 100% correct, with a mean
  error of about 0.05% of full scale.
- Alert recall is lower for the agent (89.5%) than for single pass (93.6%), because the agent declines to
  alert on values it could not confirm; single pass raised 4 false alerts against the agent's 1.

Per-condition tables are in `eval/results/results.md`. Outcomes over the 660 runs: 417 recorded, 122
warnings sent, 76 critical alerts held for approval, 43 recapture requests, 2 escalations. Mean cost per
image on a laptop CPU: 7.1 tool calls and 0.30 s (95th percentile 0.77 s).

**Failure cases.** `eval/results/agent_failures.jpg` shows the images where the agent recorded a wrong value:
3 occlusion images and 1 combined image (glare plus blur on a tilted dial), with errors of 2.3% to 3.6% of
full scale. In the occlusion cases a bar crosses the lower part of the dial near the ends of the scale, so
the rim and scale ends are estimated from what is left; the result is slightly off but still looks
self-consistent, and the confirming second reading repeats the mistake because it comes from the same
photo. The exact mechanism has not been analysed further. Glare remains the weakest condition
for coverage (70% reported): when a reflection covers the needle no processing recovers it, and the agent
correctly asks for a new photo.

### 6.1 Real photographs

**Data.** 14 openly licensed photographs from Wikimedia Commons (licences and authors in
`eval/real/SOURCES.md`). Nine show a dial a person can read; we read their values by eye (approximate,
tolerance stated per photo in `eval/real/labels.json`). Five cannot be read by a person: four gauges are
small or distant in the frame, and one antique gauge has a symmetric needle whose pointer end is ambiguous.
For those, the only correct outcome is to withhold a value. Reproduce with `python eval/run_real.py`.

| | Readable photos (9): correct | Readable photos: wrong value reported | Unreadable photos (5): value reported anyway |
|---|---|---|---|
| Single pass | 4 | **5** | 4 |
| **Agent** | 0 | **0** | **0** |

What this shows:

- **The reader does not yet transfer to real gauges.** A dial is located on 13 of 14 photos, but the
  confidence of every reading on a readable photo is below 0.1. The reasons are visible in the confidence
  components: the pivot check fails on 8 of 9 readable photos (real hubs, screws and needle bases look
  different from the synthetic ones); the tick-ring search fails on 7 of 9, including every dual-scale dial
  (bar and psi, kPa and psi), giving a sweep that does not match the asset; and real needles have counterweight tails, which the reader sometimes
  follows (pool gauge: 2.2 bar read for 0).
- **The safety design holds.** Because confidence is honest, the agent recorded nothing it could not
  trust: no wrong values and no values on unreadable photos. Every run ended in a recapture request.
  Single pass, which always reports, gave a wrong value on 5 of the 9 readable photos (errors up to 73% of
  full scale) and a value on 4 of the 5 unreadable ones.
- The four photos single pass read correctly (errors 0.2 to 1.4% of full scale) show that the geometry
  pipeline works on real dials when the scale is a single ring; the confidence terms are the part that
  needs real-data calibration.

`eval/results/real_overlays.jpg` shows each photo with the detected rim, pivot, needle and scale ends.

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

- **The reader is not yet reliable on real photographs** (section 6.1). On 14 real photos the agent
  withheld every value; the accuracy figures in section 6 apply to synthetic images only. Real gauges have
  printed logos, dual scales, coloured arcs, counterweighted needles, worn faces, curved glass and shadows
  that the renderer does not model, and the confidence thresholds were calibrated on synthetic images.
- **Small model-planner evaluation.** The 660-image numbers use the rule planner. The model planners were
  compared live on 14 gauges, one run each (section 6.2); GLM-5 takes about 15 to 20 s per run against
  0.3 s for the rules.
- **One gauge type.** A single needle on a circular dial with a tick scale and a visible gap. No dual-needle,
  linear, digital or multi-scale gauges.
- **Range comes from a registry.** The digits are not read; the asset record supplies minimum, maximum and
  nominal sweep. Selecting the wrong asset gives a wrong value with high confidence.
- **The second reading is not independent evidence.** It re-processes the same photo, so it catches
  unstable readings, not a photo that is misleading in itself (the occlusion failures above).
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

Make the reader work on real photographs: a pivot model that accepts real hubs, tick-ring selection that
handles dual scales, telling the pointer from the counterweight tail, and confidence calibrated on a real
labelled set larger than 14 photos. Run the model planners on the full 660-image set and the real photos; read the scale digits to remove the
registry dependency; take a true second photo for verification when a camera can be re-triggered.
