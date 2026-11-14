import json
import pandas as pd
import run_c100_revision_analyses as c
import address_review75 as r
r.compile_compliance()
r.compile_shape_baseline()
r.source_inference()
pred = pd.read_csv(c.OUT / 'all_terminal_predictions.csv')
summary = json.loads((c.OUT / 'analysis_summary.json').read_text())
summary['conformal'] = c.conformal_summary(pred)
(c.OUT / 'analysis_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary['conformal'], indent=2))
