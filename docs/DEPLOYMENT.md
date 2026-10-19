# Deployment notes

## Status

| Item | Status |
|---|---|
| Code, tests, evaluation, local web endpoint | Run and verified on Windows 11, Python 3.14, OpenCV 5.0.0 |
| `infra/build_lambda.py` | Run: builds a 171 MiB arm64 package (limit 250 MiB) on Windows without Docker |
| `infra/template.yaml` | **Deployed** to ap-southeast-2 (Sydney) on 15 October 2026. Health, sample, token check, a full rule-planner inspection, DynamoDB run storage and S3 evidence images verified through the live URL. |
| Model planner | **Verified live** on 20 October 2026 with GLM-5 (`zai.glm-5`) through the Converse API, the deployed default. Claude is not available on the account (AWS Free plan). |
| COOL on Graviton | Not attempted. See "COOL path" below. |

## Prerequisites

- AWS account and the AWS CLI with credentials (`aws login` or `aws configure`) and a default region.
  Docker and the SAM CLI are not needed; CloudFormation expands the SAM transform server-side.
- Bedrock model access for the model in `BedrockModelId` in the deploy region.
- The default planner calls `PlannerModel` (GLM-5) through the Bedrock Converse API; the function role
  is granted `bedrock:InvokeModel`. The optional Claude planner uses the Anthropic SDK's
  `AnthropicBedrockMantle` client and `bedrock-mantle:CreateInference`; it needs an account that may use
  Anthropic models (not the AWS Free plan). If a Bedrock call fails, the run still completes with the rule planner and the trace shows the error.
- If your organisation uses service control policies, deploy to a region they allow. On the development
  account, S3 bucket creation was denied in ap-southeast-1 and allowed in ap-southeast-2.

## Deploy

From the repository root:

```bash
python infra/build_lambda.py                      # Linux arm64 wheels + gaugeguard -> infra/build/
aws s3 mb s3://<artifact-bucket>                  # once; any unique name, holds the uploaded zip
aws cloudformation package --template-file infra/template.yaml --s3-bucket <artifact-bucket> \
  --output-template-file infra/packaged.yaml
aws cloudformation deploy --template-file infra/packaged.yaml --stack-name gaugeguard \
  --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND \
  --parameter-overrides Planner=bedrock DemoToken=<token> AlertEmail=<address>
aws cloudformation describe-stacks --stack-name gaugeguard --query "Stacks[0].Outputs"
```

Then open the `Url` output. Confirm the SNS subscription e-mail to receive alerts.

Smoke test:

```bash
curl <Url>api/health
```

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `GG_PLANNER` | `rules` locally, `bedrock` in the template | Planner: `rules`, `bedrock` (Converse API) or `claude` |
| `GG_PLANNER_MODEL` | `zai.glm-5` | Bedrock model for the `bedrock` planner (template parameter `PlannerModel`) |
| `GG_BEDROCK_MODEL` | `anthropic.claude-opus-5-5` | Bedrock model id |
| `GG_BEDROCK_REGION` | the function's region | Region for Bedrock calls (template parameter `BedrockRegion`), for deploy regions where the model is not offered |
| `GG_MIN_CONFIDENCE` | `0.6` | Minimum confidence to record a reading |
| `GG_AGREE_FRACTION` | `0.02` | Two readings agree within this fraction of full scale |
| `GG_MAX_STEPS` | `20` | Tool-call budget per run |
| `GG_DEMO_TOKEN` | unset | When set, inspect and decision requests need `Authorization: Bearer <token>` |
| `GG_TABLE`, `GG_BUCKET`, `GG_ALERT_TOPIC` | unset | Set by the template; unset means local file storage |

## Security and responsible operation

- The Function URL is public so judges can open it. Set `DemoToken` so only people with the token can run
  inspections or approve alerts. Reserved concurrency of 5 caps cost.
- Uploads are limited to 6 MB, decoded with OpenCV, and downscaled to 960 px. Function URL requests are
  capped at 6 MB by AWS, so photos above roughly 4 MB must be resized by the client.
- Photos are stored privately, encrypted, and deleted after 30 days. Run records expire after 30 days.
- Photos are never sent to the model; the planner sees only numeric tool results.
- Critical alerts are held until a named operator approves them; the decision is stored with the run.
- Metrics (`GaugeGuard` namespace: Runs, Steps, DurationMs, Confidence by Outcome) come from structured
  log lines. Two alarms are created: function errors, and a high rate of escalations.

## Cost (rough guide, verify with the AWS calculator)

Lambda, DynamoDB, S3 and SNS usage for a demo is within the Free Tier. The variable cost is Bedrock: one
run is typically 5 to 10 model turns over a short conversation. Set `Planner=rules` for zero model cost.

## Teardown

```bash
aws s3 rm s3://<Bucket output> --recursive
aws cloudformation delete-stack --stack-name gaugeguard
aws s3 rb s3://<artifact-bucket> --force
```

## COOL path (optional, for the Best Use of COOL Award)

The function already targets arm64 (Graviton). The operations GaugeGuard spends its time in are ones COOL
is described as accelerating (resize, Gaussian and adaptive thresholding, contour extraction). To make a
COOL claim you would need to:

1. Subscribe to COOL on AWS Marketplace and build the Lambda image (or an ECS/EC2 Graviton service) on it,
   replacing `opencv-python-headless` in `requirements.txt`.
2. Run `python bench/bench_ops.py --out stock.json` on stock OpenCV 5 and `--out cool.json` on COOL, on the
   same Graviton instance type, and report both files.
3. Record the COOL version and instance type in the technical report.

None of this has been done yet, so the project does not currently claim the COOL path.
