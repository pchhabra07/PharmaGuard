"""
PharmaGuard — Dashboard HTML Template
=========================================
Server-side rendered HTML dashboard that showcases the model's
performance metrics, historical runs, and API usage examples.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def render_dashboard(config: dict, metrics_path: Path, base_url: str = "") -> str:
    # Load metrics history
    history = []
    latest = None
    try:
        metrics_path = Path(metrics_path)
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            if history:
                latest = history[-1]
    except Exception as exc:
        logger.warning("Could not load metrics for dashboard: %s", exc)

    train_cfg = config.get("training", {})
    api_cfg = config.get("api", {})
    model_version = api_cfg.get("model_version", "v1")

    # Extract latest metrics
    if latest:
        test_m = latest.get("metrics", {}).get("test", {})
        precision = test_m.get("precision", 0)
        recall = test_m.get("recall", 0)
        f1 = test_m.get("f1", 0)
        roc_auc = test_m.get("roc_auc", 0)
        pr_auc = test_m.get("pr_auc", 0)
        accuracy = test_m.get("accuracy", 0)
        threshold = latest.get("threshold", train_cfg.get("threshold", 0.5))
    else:
        precision = recall = f1 = roc_auc = pr_auc = accuracy = 0
        threshold = train_cfg.get("threshold", 0.5)

    history_rows = _build_history_rows(history)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PharmaGuard - Adverse Drug Reaction Detection Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after {{ margin: 0; padding: 0; box-sizing: border-box; }}
        :root {{
            --bg-primary: #0a0a1a;
            --bg-card: rgba(255, 255, 255, 0.04);
            --bg-card-hover: rgba(255, 255, 255, 0.07);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-cyan: #06b6d4;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-pink: #ec4899;
            --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --gradient-header: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        }}
        body {{
            font-family: 'Inter', sans-serif; background: var(--bg-primary); color: var(--text-primary); line-height: 1.6; min-height: 100vh;
        }}
        body::before {{
            content: ''; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(ellipse at 20% 50%, rgba(102,126,234,0.08) 0%, transparent 50%),
                        radial-gradient(ellipse at 80% 20%, rgba(118,75,162,0.06) 0%, transparent 50%);
            pointer-events: none; z-index: 0;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 0 24px; position: relative; z-index: 1; }}
        
        /* Header */
        .header {{ background: var(--gradient-header); border-bottom: 1px solid var(--border-subtle); padding: 48px 0 56px; text-align: center; position: relative; }}
        .header h1 {{ font-size: 42px; font-weight: 800; background: linear-gradient(135deg, #e2e8f0, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 12px; }}
        .header p {{ font-size: 17px; color: var(--text-secondary); max-width: 600px; margin: 0 auto 24px; }}
        
        .section {{ padding: 48px 0; }}
        .section-title {{ font-size: 22px; font-weight: 700; margin-bottom: 8px; }}
        .section-subtitle {{ font-size: 14px; color: var(--text-secondary); margin-bottom: 32px; }}
        
        /* Metrics Grid */
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 16px; }}
        .metric-card {{ background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 14px; padding: 24px 20px; transition: all 0.3s ease; position: relative; overflow: hidden; }}
        .metric-card:hover {{ background: var(--bg-card-hover); transform: translateY(-2px); }}
        .metric-card::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 14px 14px 0 0; }}
        .metric-card--blue::before {{ background: var(--accent-blue); }} .metric-card--purple::before {{ background: var(--accent-purple); }} .metric-card--cyan::before {{ background: var(--accent-cyan); }}
        .metric-card--emerald::before {{ background: var(--accent-emerald); }} .metric-card--amber::before {{ background: var(--accent-amber); }} .metric-card--pink::before {{ background: var(--accent-pink); }}
        .metric-label {{ font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 8px; }}
        .metric-value {{ font-size: 32px; font-weight: 800; }}
        .metric-detail {{ font-size: 12px; color: var(--text-muted); margin-top: 4px; }}
        
        /* Forms & Inputs */
        .form-card {{ background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 14px; padding: 28px; backdrop-filter: blur(12px); }}
        .form-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .input-group {{ display: flex; flex-direction: column; gap: 6px; }}
        .input-group label {{ font-size: 13px; font-weight: 600; color: var(--text-secondary); }}
        .input-group input, .input-group select {{ background: rgba(0,0,0,0.2); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 10px 14px; color: white; font-family: inherit; font-size: 14px; outline: none; transition: border-color 0.2s; }}
        .input-group input:focus, .input-group select:focus {{ border-color: var(--accent-blue); }}
        .btn {{ background: var(--gradient-primary); color: white; border: none; padding: 12px 24px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; width: 100%; font-size: 15px; }}
        .btn:hover {{ transform: translateY(-1px); box-shadow: 0 4px 15px rgba(102,126,234,0.4); }}
        
        /* Prediction Result */
        .result-box {{ margin-top: 24px; padding: 20px; border-radius: 10px; display: none; align-items: center; gap: 16px; border: 1px solid; animation: fadeInUp 0.4s ease; }}
        .result-box.serious {{ background: rgba(239, 68, 68, 0.1); border-color: rgba(239, 68, 68, 0.3); color: #fca5a5; }}
        .result-box.safe {{ background: rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.3); color: #6ee7b7; }}
        
        /* History Table & Plots */
        .table-wrapper {{ background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 14px; overflow: hidden; backdrop-filter: blur(12px); }}
        .table-wrapper table {{ width: 100%; border-collapse: collapse; }}
        .table-wrapper th {{ background: rgba(255, 255, 255, 0.03); padding: 14px 16px; font-size: 11px; text-transform: uppercase; color: var(--text-secondary); text-align: left; }}
        .table-wrapper td {{ padding: 12px 16px; font-size: 13px; border-bottom: 1px solid var(--border-subtle); cursor: pointer; }}
        .table-wrapper tr:hover td {{ background: rgba(255, 255, 255, 0.05); }}
        .badge {{ padding: 3px 10px; border-radius: 100px; font-size: 11px; font-weight: 600; }}
        .badge--latest {{ background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); }}
        
        .plot-container {{ display: none; padding: 24px; background: rgba(0,0,0,0.3); border-bottom: 1px solid var(--border-subtle); animation: slideDown 0.3s ease; }}
        .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .plot-grid img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--border-subtle); background: white; }}
        
        @keyframes fadeInUp {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        @keyframes slideDown {{ from {{ opacity: 0; max-height: 0; }} to {{ opacity: 1; max-height: 1000px; }} }}
    </style>
</head>
<body>
    <header class="header">
        <div class="container">
            <h1>PharmaGuard Dashboard</h1>
            <p>Real-time ADR severity predictions powered by XGBoost. Serving version <strong style="color:var(--text-primary)">{model_version}</strong>.</p>
        </div>
    </header>

    <section class="section">
        <div class="container">
            <h2 class="section-title">Live Prediction</h2>
            <p class="section-subtitle">Test the `/predict` API endpoint directly from the browser.</p>
            <div class="form-card">
                <form id="predict-form" onsubmit="event.preventDefault(); makePrediction();">
                    <div class="form-grid">
                        <div class="input-group">
                            <label>Age Group</label>
                            <select id="f_age" required>
                                <option value="18-44">18-44</option>
                                <option value="0-17">0-17</option>
                                <option value="45-64">45-64</option>
                                <option value="65-74">65-74</option>
                                <option value="75+">75+</option>
                            </select>
                        </div>
                        <div class="input-group">
                            <label>Sex</label>
                            <select id="f_sex" required>
                                <option value="M">Male (M)</option>
                                <option value="F">Female (F)</option>
                            </select>
                        </div>
                        <div class="input-group">
                            <label>Route</label>
                            <input type="text" id="f_route" value="Oral" required>
                        </div>
                        <div class="input-group">
                            <label>Drug Name</label>
                            <input type="text" id="f_drug" value="ASPIRIN" required>
                        </div>
                        <div class="input-group">
                            <label>Polypharmacy Count</label>
                            <input type="number" id="f_poly" value="3" min="1" required>
                        </div>
                    </div>
                    <button type="submit" class="btn" id="btn-predict">Predict Severity</button>
                </form>
                
                <div id="result-box" class="result-box">
                    <div style="font-size: 32px;" id="res-icon"></div>
                    <div>
                        <h3 id="res-title" style="font-size: 18px; font-weight: 700; margin-bottom: 4px;"></h3>
                        <p id="res-desc" style="font-size: 14px; opacity: 0.9;"></p>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <section class="section" style="padding-top: 0;">
        <div class="container">
            <h2 class="section-title">Latest Model Performance</h2>
            <p class="section-subtitle">Evaluation on held-out test set for {model_version}</p>
            <div class="metrics-grid">
                <div class="metric-card metric-card--blue"><div class="metric-label">Precision</div><div class="metric-value" style="color:var(--accent-blue)">{precision:.2%}</div></div>
                <div class="metric-card metric-card--purple"><div class="metric-label">Recall</div><div class="metric-value" style="color:var(--accent-purple)">{recall:.2%}</div></div>
                <div class="metric-card metric-card--cyan"><div class="metric-label">F1 Score</div><div class="metric-value" style="color:var(--accent-cyan)">{f1:.2%}</div></div>
                <div class="metric-card metric-card--emerald"><div class="metric-label">ROC-AUC</div><div class="metric-value" style="color:var(--accent-emerald)">{roc_auc:.2%}</div></div>
                <div class="metric-card metric-card--pink"><div class="metric-label">PR-AUC</div><div class="metric-value" style="color:var(--accent-pink)">{pr_auc:.2%}</div></div>
            </div>
        </div>
    </section>

    <section class="section" style="padding-top: 0;">
        <div class="container">
            <h2 class="section-title">Training History & Plots</h2>
            <p class="section-subtitle">Click on any run below to view its evaluation plots.</p>
            {_render_empty_or_table(history, history_rows)}
        </div>
    </section>

    <script>
        const baseUrl = "{base_url}";

        async function makePrediction() {{
            const btn = document.getElementById('btn-predict');
            const resBox = document.getElementById('result-box');
            
            btn.textContent = "Predicting...";
            btn.style.opacity = 0.7;
            
            const payload = {{
                age_group: document.getElementById('f_age').value,
                sex: document.getElementById('f_sex').value,
                route: document.getElementById('f_route').value,
                drug_name_normalized: document.getElementById('f_drug').value,
                polypharmacy_count: parseInt(document.getElementById('f_poly').value)
            }};

            try {{
                const res = await fetch(baseUrl + '/predict', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify(payload)
                }});
                const data = await res.json();
                
                resBox.style.display = 'flex';
                if(data.is_serious) {{
                    resBox.className = 'result-box serious';
                    document.getElementById('res-icon').innerHTML = '⚠️';
                    document.getElementById('res-title').innerText = 'High Risk of Serious ADR';
                }} else {{
                    resBox.className = 'result-box safe';
                    document.getElementById('res-icon').innerHTML = '✅';
                    document.getElementById('res-title').innerText = 'Low Risk';
                }}
                document.getElementById('res-desc').innerText = `Probability: ${{(data.probability*100).toFixed(2)}}% | Threshold: ${{data.threshold}} | Model: ${{data.model_version}}`;
                
            }} catch(e) {{
                alert("Prediction failed. Check console for details.");
                console.error(e);
            }} finally {{
                btn.textContent = "Predict Severity";
                btn.style.opacity = 1;
            }}
        }}

        function togglePlots(rowId, version) {{
            const plotRow = document.getElementById('plots-' + rowId);
            if(plotRow.style.display === 'table-row') {{
                plotRow.style.display = 'none';
            }} else {{
                // Hide all other plot rows
                document.querySelectorAll('.plot-row').forEach(r => r.style.display = 'none');
                plotRow.style.display = 'table-row';
                
                // Lazy load images
                const container = document.getElementById('plot-container-' + rowId);
                if(container.innerHTML.trim() === '') {{
                    container.innerHTML = `
                        <div class="plot-grid">
                            <div><p style="margin-bottom:8px;font-size:12px;color:#94a3b8">ROC Curve</p><img src="${{baseUrl}}/plots/${{version}}/test_roc_curve.png" onerror="this.parentElement.style.display='none'"></div>
                            <div><p style="margin-bottom:8px;font-size:12px;color:#94a3b8">PR Curve</p><img src="${{baseUrl}}/plots/${{version}}/test_pr_curve.png" onerror="this.parentElement.style.display='none'"></div>
                            <div><p style="margin-bottom:8px;font-size:12px;color:#94a3b8">Confusion Matrix</p><img src="${{baseUrl}}/plots/${{version}}/test_confusion_matrix.png" onerror="this.parentElement.style.display='none'"></div>
                            <div><p style="margin-bottom:8px;font-size:12px;color:#94a3b8">Feature Importance</p><img src="${{baseUrl}}/plots/${{version}}/test_feature_importance.png" onerror="this.parentElement.style.display='none'"></div>
                        </div>
                    `;
                }}
            }}
        }}
    </script>
</body>
</html>"""


def _build_history_rows(history: list) -> str:
    if not history: return ""
    rows = []
    for i, run in enumerate(reversed(history)):
        test_m = run.get("metrics", {}).get("test", {})
        is_latest = i == 0
        badge = ' <span class="badge badge--latest">latest</span>' if is_latest else ""
        timestamp = run.get("timestamp", "N/A").split("T")[0]
        version = run.get("version", "v1")
        
        # Main row
        rows.append(f'''
            <tr onclick="togglePlots({i}, '{version}')">
                <td>{version}{badge}</td>
                <td>{timestamp}</td>
                <td>{run.get("threshold", "N/A")}</td>
                <td>{test_m.get("precision", 0):.4f}</td>
                <td>{test_m.get("recall", 0):.4f}</td>
                <td>{test_m.get("f1", 0):.4f}</td>
                <td>{test_m.get("roc_auc", 0):.4f}</td>
            </tr>
            <tr id="plots-{i}" class="plot-row" style="display:none;">
                <td colspan="7" style="padding:0; border:none;">
                    <div id="plot-container-{i}" class="plot-container" style="display:block;"></div>
                </td>
            </tr>
        ''')
    return "\\n".join(rows)


def _render_empty_or_table(history: list, rows_html: str) -> str:
    if not history: return "<p>No training history yet.</p>"
    return f"""
    <div class="table-wrapper">
        <table>
            <thead>
                <tr>
                    <th>Version</th>
                    <th>Date</th>
                    <th>Threshold</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>ROC-AUC</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
