# Submission checklist

Deadline: **26 October 2026, 11:45 pm PDT** (Devpost page; the overview text says 11:59 pm, so use the
earlier time). Rules: https://opencv26.devpost.com/rules

## Required items

| Requirement | Status | Where / what is left |
|---|---|---|
| Technical report (problem, users, architecture, OpenCV 5, AWS, evaluation, limitations, responsible use) | Drafted | `docs/TECHNICAL_REPORT.md`. Add team details and links; update sections 5 and 7 after deploying. |
| Judge-accessible code repository or archive | To do | Push this folder to a Git host, or zip it (exclude `.venv`). |
| Pinned dependencies and build, deploy, test instructions | Done | `requirements.txt`, `README.md`, `docs/DEPLOYMENT.md` |
| Architecture diagram with OpenCV 5, AWS and agent components | Done | `README.md` (Mermaid). Export as an image for the report if the form needs one. |
| Working web endpoint, or an arranged live screen-share | Done (rule planner) | https://u7vijv2bb2ouird3dz6ktymsn40jybjv.lambda-url.ap-southeast-2.on.aws/ (Sydney). Give judges the demo token. Bedrock planner waits on Claude model access for the account. |
| Video, 5 minutes or less, showing the team, the app working, architecture and results | To do | Script in `docs/VIDEO_SCRIPT.md` |
| Evaluation evidence including failure cases | Done (synthetic) | `eval/results/` |

## Agentic Vision Award evidence

| Evidence | Status | Where |
|---|---|---|
| Agent workflow diagram (perception, decision, action) | Done | `README.md` |
| Trace showing OpenCV 5 output changing a later decision | Done | `eval/results/example_traces.json`; live in the UI trace panel |
| Evaluation of task success, failure handling, observability, human control | Done (rule planner) | Technical report sections 4 and 6 |

Opt in to the Agentic Vision Award on the submission form. Do not opt in to the COOL award unless the COOL
steps in `docs/DEPLOYMENT.md` are completed.

## Before you submit

1. Deploy to AWS and confirm one full run with the Bedrock planner. Save that trace.
2. Photograph several real gauges, add matching entries to `gaugeguard/assets.json`, and record how the
   reader does. Add the results, good or bad, to the report. This is the largest gap in the evidence.
3. Record the video.
4. Re-run `python -m pytest -q` and `python eval/run_eval.py --n 60` and make sure the numbers in the report
   still match `eval/results/results.md`.
5. Check eligibility: every team member meets the age rule, and one member is the named representative.
