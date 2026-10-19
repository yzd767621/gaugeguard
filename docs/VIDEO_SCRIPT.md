# Video script (target 4 min 30 s, limit 5 min)

| Time | On screen | Say |
|---|---|---|
| 0:00 to 0:25 | Team on camera | Who we are. Plants still read analog gauges by hand. A photo reader is only useful if it never quietly reports a wrong number. |
| 0:25 to 1:05 | UI: generate a **clean** image, press Inspect | GaugeGuard reads the gauge with OpenCV 5. Point at the overlay: rim, needle, scale ends. Five steps, value recorded with its confidence. |
| 1:05 to 1:50 | UI: **low light** image, show the trace | First reading has low confidence. The quality tool measured "dark", so the agent chose to denoise, located the dial again and re-read. Confidence is now high. The visual evidence changed the next tool call. |
| 1:50 to 2:30 | UI: an image above the critical limit | The value is over the limit, so the agent takes a second reading before acting. They agree, so it records, but the critical alert is held. Type a name, approve. Nothing is sent without a person. |
| 2:30 to 2:55 | UI: **glare** image | Processing cannot recover a needle hidden by a reflection. The agent stops and tells the technician what to change. |
| 2:55 to 3:35 | Architecture diagram | Lambda on Graviton runs OpenCV 5 and the agent loop. A model on Bedrock (GLM-5) plans from tool results and never sees the photo. DynamoDB, S3, SNS, CloudWatch. Guardrails are in code, so they hold for any planner. |
| 3:35 to 4:15 | Results table | 660 unseen test images. Always reporting gives a wrong value on 5.0%. The agent gives a correct value on 92.6% and a wrong one on 0.6%. Show the failure image: a bar hiding the end of the scale. |
| 4:15 to 4:30 | Limitations slide | Synthetic images so far; real photos are the next step. Decision support, not a safety system. |

Before recording: replace the results line if the numbers change, and show the live AWS URL in the browser
bar if deployed.
