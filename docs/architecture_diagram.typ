#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#set page(width: auto, height: auto, margin: 5mm, fill: white)
#set text(font: "New Computer Modern", size: 9pt)

// Color palette
#let gcp-blue = rgb("#4285F4")
#let gcp-green = rgb("#34A853")
#let gcp-red = rgb("#EA4335")
#let gcp-yellow = rgb("#FBBC04")
#let gray-light = rgb("#F1F3F4")
#let gray-med = rgb("#DADCE0")
#let dark-text = rgb("#202124")
#let blue-fill = gcp-blue.lighten(85%)
#let green-fill = gcp-green.lighten(85%)
#let red-fill = gcp-red.lighten(90%)
#let yellow-fill = gcp-yellow.lighten(85%)
#let orange-fill = rgb("#FFF3E0")

// Node helper
#let gcp-node(pos, title, subtitle, tint, w: 3.2cm) = node(
  pos, align(center)[#text(weight: "bold", size: 8.5pt, title) #linebreak() #text(size: 7.5pt, subtitle)],
  width: w, height: 1.1cm, fill: tint, stroke: .8pt + tint.darken(30%), corner-radius: 3pt,
)

#diagram(
  spacing: (1.2cm, 1.4cm),
  edge-stroke: .6pt + dark-text,
  edge-corner-radius: 4pt,
  mark-scale: 65%,

  // ===== TOP ROW: External + Serving =====
  gcp-node((0, 0), "GitHub", "Repository", gray-light, w: 2.4cm),
  gcp-node((2, 0), "Cloud Build", "CI/CD", blue-fill, w: 2.6cm),
  gcp-node((4, 0), "Artifact Registry", "Docker Images", orange-fill, w: 3cm),
  gcp-node((6.5, 0), "Cloud Run", "Serving API", blue-fill, w: 2.6cm),
  gcp-node((9, 0), "Synthetic API", "nomic-embed-text-v1.5", gray-light, w: 3.2cm),
  gcp-node((12, 0), "Cloud Monitoring", "Alerts + Logs", orange-fill, w: 2.8cm),

  // ===== CLIENT (above Cloud Run) =====
  gcp-node((6.5, -2), "Client", "/predict, /health", gray-light, w: 2.4cm),

  // ===== MIDDLE ROW: Data + Training + Retraining =====
  gcp-node((0, -3.5), "Cloud Scheduler", "cron semanal", green-fill, w: 2.6cm),
  gcp-node((2.5, -3.5), "Pub/Sub", "retrain-trigger", green-fill, w: 2.4cm),
  gcp-node((5, -3.5), "Cloud Function", "trigger-retraining", green-fill, w: 3cm),
  gcp-node((8.5, -3.5), "Cloud Storage", "gs://mlops-toxic-classifier-ml", yellow-fill, w: 4.5cm),
  gcp-node((13, -3.5), "Vertex AI", "Custom Training Job", blue-fill, w: 3.2cm),

  // ===== BOTTOM: Secret Manager =====
  gcp-node((8.5, -5.2), "Secret Manager", "synthetic-api-key", red-fill, w: 3cm),

  // ===== EDGES =====

  // CI/CD flow
  edge((0, 0), (2, 0), "-|>", label: [push]),
  edge((2, 0), (4, 0), "-|>", label: [build+push]),
  edge((4, 0), (6.5, 0), "-|>", label: [image]),

  // Client -> Cloud Run
  edge((6.5, -2), (6.5, 0), "-|>", label: [HTTP], label-pos: .3),

  // Cloud Run -> Synthetic API
  edge((6.5, 0), (9, 0), "-|>", label: [embeddings]),

  // Cloud Run -> GCS (model loading)
  edge((6.5, 0), (8.5, -3.5), "-|>", label: [model at startup], label-pos: .4),

  // AR -> Vertex AI (image)
  edge((4, 0), (13, -3.5), "-|>", label: [image], label-pos: .7),

  // GCS <-> Vertex AI
  edge((8.5, -3.5), (13, -3.5), "-|>", label: [data + cache]),
  edge((13, -3.5), (10.5, -3.5), "<|-", label: [write model/], label-pos: .3),

  // Secrets -> Vertex AI
  edge((8.5, -5.2), (13, -4.7), "-|>", label: [API key]),

  // Retraining flow
  edge((0, -3.5), (2.5, -3.5), "-|>", label: [trigger]),
  edge((2.5, -3.5), (5, -3.5), "-|>", label: [event]),
  edge((5, -3.5), (13, -3.5), "-|>", label: [launch job]),

  // Monitoring
  edge((6.5, 0), (12, 0), "-|>", label: [logs], label-pos: .7),
  edge((13, -3.5), (12, -1.5), "-|>", label: [job status], label-pos: .6),
)
