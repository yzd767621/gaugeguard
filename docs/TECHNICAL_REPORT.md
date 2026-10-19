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
   Two planners share one interface: Claude on Amazon Bedrock, and a deterministic rule planner.
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

**Model planner.** `bedrock.py` uses the Anthropic SDK's Bedrock client with strict tool schemas and adaptive
thinking. The model never receives the photo, only tool results, which keeps images inside the account and
makes every decision traceable to a measurement. If the model is unreachable, throttled or declines, the
rule planner takes over from the current state and the trace records the hand-over.

**Observability.** Every run stores a trace of each tool call with its arguments, the planner's stated
reason, the result and the time taken. Runs emit CloudWatch metrics by outcome through structured logs.

## 5. AWS deployment

One arm64 (Graviton) Lambda container behind a Function URL serves the web page and the API and runs the
agent loop; Amazon Bedrock supplies the planner; DynamoDB stores runs and readings; S3 stores photos and
annotated evidence; SNS delivers alerts; CloudWatch holds logs, metrics and two alarms. The stack is defined
in `infra/template.yaml` and deploys with `infra/build_lambda.py` and `aws cloudformation package/deploy`. See [DEPLOYMENT.md](DEPLOYMENT.md).

**Status: the deployment files are written but have not yet been deployed or tested against a live AWS
account, and the Bedrock planner has not yet been run against a live model.** Both must be done before
submission; the local server exercises the same request handler the Lambda uses.

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

## 7. Limitations

- **Synthetic evaluation only.** No real photographs have been tested. Real gauges have printed logos,
  mirror bands, coloured arcs, worn faces, curved glass and shadows that the renderer does not model.
  Quality thresholds were calibrated on synthetic images and will need re-tuning on real ones.
- **No live model evaluation.** The reported agent numbers use the rule planner. The Bedrock planner is
  tested only with a scripted client.
- **One gauge type.** A single needle on a circular dial with a tick scale and a visible gap. No dual-needle,
  linear, digital or multi-scale gauges.
- **Range comes from a registry.** The digits are not read; the asset record supplies minimum, maximum and
  nominal sweep. Selecting the wrong asset gives a wrong value with high confidence.
- **The second reading is not independent evidence.** It re-processes the same photo, so it catches
  unstable readings, not a photo that is misleading in itself (the occlusion failures above).
- **Linear scales only**, and the first and last ticks are assumed to be the scale's minimum and maximum.
- **Deployment not yet exercised** (section 5), and the COOL path has not been attempted.

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
- **Data rights.** All evaluation images are generated by this project's own code; no third-party images
  or datasets are used.
- **Security.** Optional bearer token on write endpoints, input size limits, reserved concurrency,
  least-privilege IAM policies scoped to the stack's own resources.

## 9. Next steps

Collect and label real gauge photos and repeat the evaluation; run the Bedrock planner on the same set and
compare it with the rule planner; deploy and record the live endpoint; read the scale digits to remove the
registry dependency; take a true second photo for verification when a camera can be re-triggered.
