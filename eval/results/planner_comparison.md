# Model planners compared (live Bedrock runs, 20 October 2026)

Command: `python eval/compare_planners.py zai.glm-5@us-east-1,moonshotai.kimi-k2.5@us-east-1,openai.gpt-oss-120b-1:0@us-east-1,deepseek.v3.2@us-east-1,amazon.nova-pro-v1:0`

14 unseen synthetic gauges (seeds from 8000): one per condition (clean, low light, glare, noise, combined,
motion, occlusion, tilt, blur, JPEG) plus 6 with values outside the asset's limits, where the agent must
verify the value before alerting. Every planner drives the same tools under the same code-enforced
guardrails. A reading is correct within 2% of full scale.

| Planner | Correct | Wrong value recorded | Needed alerts raised | Mean tool calls | Mean time per run | Tokens per run | Steps without an explanation |
|---|---|---|---|---|---|---|---|
| Rules (deterministic) | 14/14 | 0 | 6/6 | 7.8 | 0.3 s | 0 | 0 |
| **GLM-5** (`zai.glm-5`) | 13/14 | 0 | 5/6 | 8.8 | 18.7 s | 16,872 | 0 |
| DeepSeek V3.2 | 13/14 | 0 | 5/6 | 8.7 | 19.5 s | 18,942 | 0 |
| gpt-oss-120b | 12/14 | 0 | 5/6 | 7.9 | 11.7 s | 12,941 | 54 |
| Kimi K2.5 | 11/14 | 0 | 3/6 | 7.7 | 14.8 s | 11,833 | 0 |
| Amazon Nova Pro | 11/14 | 0 | 4/6 | 7.6 | 9.5 s | 15,475 | 0 |

Findings:

- **No model planner recorded a wrong value.** The guardrails are in the tools, so a planner that is unsure
  can only withhold (recapture or escalation) rather than record a value the evidence does not support.
- Model planners are more cautious than the rules: their misses are recapture requests or escalations,
  for example GLM-5 asked for a new photo on one critical-low gauge where the rules confirmed and alerted.
- Model planners sometimes do better than the rules by trying more remedies: on a combined-degradation and a
  motion-blurred image GLM-5 recorded 0.03% and 0.14% of full-scale error where the rules recorded 0.31% and 0.34%.
- GLM-5 was chosen as the default model planner: best accuracy (tied with DeepSeek V3.2) with fewer tokens,
  and it explains every step for the audit trail. gpt-oss-120b left 54 steps without an explanation.
- Claude models are not used in this comparison because the AWS account is on the Free plan, which does not
  allow Anthropic models; the Claude planner remains in the code (planner `claude`).

Small sample (14 gauges, one run each), so differences of one or two gauges are not significant.
