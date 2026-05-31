"""
PharmaGuard — Dashboard HTML Template
=========================================
Server-side rendered HTML dashboard that showcases the model's
performance metrics, historical runs, and API usage examples.

Charts are rendered dynamically on the client with Chart.js,
fetching data from the ``/model/charts`` API endpoint.
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
        specificity = test_m.get("specificity", 0)
        threshold = latest.get("threshold", train_cfg.get("threshold", 0.5))
        n_estimators = latest.get("config", {}).get("n_estimators", train_cfg.get("n_estimators", 500))
        max_depth = latest.get("config", {}).get("max_depth", train_cfg.get("max_depth", 6))
        learning_rate = latest.get("config", {}).get("learning_rate", train_cfg.get("learning_rate", 0.1))
        scale_pos_weight = latest.get("config", {}).get("scale_pos_weight", train_cfg.get("scale_pos_weight", 1.0))
        eval_metric = latest.get("config", {}).get("eval_metric", train_cfg.get("eval_metric", "aucpr"))
        early_stopping = latest.get("config", {}).get("early_stopping_rounds", train_cfg.get("early_stopping_rounds", 30))
        threshold_strategy = latest.get("threshold_strategy", train_cfg.get("threshold_strategy", "f1"))
        quarters = ", ".join(latest.get("quarters", config.get("faers", {}).get("quarters", [])))
    else:
        precision = recall = f1 = roc_auc = pr_auc = accuracy = specificity = 0
        threshold = train_cfg.get("threshold", 0.5)
        n_estimators = train_cfg.get("n_estimators", 500)
        max_depth = train_cfg.get("max_depth", 6)
        learning_rate = train_cfg.get("learning_rate", 0.1)
        scale_pos_weight = train_cfg.get("scale_pos_weight", 1.0)
        eval_metric = train_cfg.get("eval_metric", "aucpr")
        early_stopping = train_cfg.get("early_stopping_rounds", 30)
        threshold_strategy = train_cfg.get("threshold_strategy", "f1")
        quarters = ", ".join(config.get("faers", {}).get("quarters", []))

    history_rows = _build_history_rows(history)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PharmaGuard — Adverse Drug Reaction Detection Dashboard</title>
    <meta name="description" content="PharmaGuard: ML-powered dashboard for predicting serious Adverse Drug Reactions using XGBoost trained on FDA FAERS data.">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
    <style>
        *, *::before, *::after {{ margin: 0; padding: 0; box-sizing: border-box; }}
        :root {{
            --bg-primary: #06070d;
            --bg-secondary: #0c0e1a;
            --bg-card: rgba(255, 255, 255, 0.03);
            --bg-card-hover: rgba(255, 255, 255, 0.06);
            --bg-card-solid: #10122a;
            --border-subtle: rgba(255, 255, 255, 0.06);
            --border-hover: rgba(255, 255, 255, 0.12);
            --text-primary: #e8edf5;
            --text-secondary: #8892a8;
            --text-muted: #5a6478;
            --accent-blue: #4f8fff;
            --accent-purple: #a855f7;
            --accent-cyan: #22d3ee;
            --accent-emerald: #34d399;
            --accent-amber: #fbbf24;
            --accent-pink: #f472b6;
            --accent-red: #f87171;
            --accent-indigo: #818cf8;
            --gradient-primary: linear-gradient(135deg, #4f8fff 0%, #a855f7 100%);
            --gradient-header: linear-gradient(160deg, #06070d 0%, #0f1129 40%, #150d2e 70%, #06070d 100%);
            --gradient-card: linear-gradient(135deg, rgba(79,143,255,0.06) 0%, rgba(168,85,247,0.04) 100%);
            --shadow-glow: 0 0 40px rgba(79, 143, 255, 0.06);
            --radius: 16px;
            --radius-sm: 10px;
            --radius-xs: 6px;
        }}
        html {{ scroll-behavior: smooth; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
            overflow-x: hidden;
        }}
        body::before {{
            content: ''; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background:
                radial-gradient(ellipse 800px 600px at 15% 30%, rgba(79,143,255,0.06) 0%, transparent 70%),
                radial-gradient(ellipse 600px 500px at 85% 15%, rgba(168,85,247,0.05) 0%, transparent 70%),
                radial-gradient(ellipse 500px 400px at 50% 80%, rgba(34,211,238,0.03) 0%, transparent 70%);
            pointer-events: none; z-index: 0;
        }}

        .container {{ max-width: 1280px; margin: 0 auto; padding: 0 28px; position: relative; z-index: 1; }}

        /* ── Scroll reveal ────────────────────────────── */
        .reveal {{
            opacity: 0; transform: translateY(30px);
            transition: opacity 0.7s cubic-bezier(0.16, 1, 0.3, 1), transform 0.7s cubic-bezier(0.16, 1, 0.3, 1);
        }}
        .reveal.visible {{ opacity: 1; transform: translateY(0); }}

        /* ── Header ───────────────────────────────────── */
        .header {{
            background: var(--gradient-header);
            border-bottom: 1px solid var(--border-subtle);
            padding: 56px 0 64px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        .header::after {{
            content: ''; position: absolute; bottom: 0; left: 50%; transform: translateX(-50%);
            width: 200px; height: 1px;
            background: linear-gradient(90deg, transparent, rgba(79,143,255,0.5), transparent);
        }}
        .header-badge {{
            display: inline-flex; align-items: center; gap: 6px;
            background: rgba(79,143,255,0.1); border: 1px solid rgba(79,143,255,0.2);
            border-radius: 100px; padding: 6px 16px; font-size: 12px; font-weight: 600;
            color: var(--accent-blue); margin-bottom: 20px; letter-spacing: 0.5px;
        }}
        .header-badge .dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--accent-emerald); animation: pulse-dot 2s ease infinite; }}
        @keyframes pulse-dot {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }} }}
        .header h1 {{
            font-size: 48px; font-weight: 900; letter-spacing: -1.5px;
            background: linear-gradient(135deg, #e8edf5 0%, #ffffff 40%, #c4b5fd 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; margin-bottom: 14px;
        }}
        .header p {{ font-size: 17px; color: var(--text-secondary); max-width: 580px; margin: 0 auto; line-height: 1.7; }}
        .header p strong {{ color: var(--text-primary); }}

        /* ── Sections ─────────────────────────────────── */
        .section {{ padding: 56px 0; }}
        .section-header {{ margin-bottom: 36px; }}
        .section-label {{
            display: inline-flex; align-items: center; gap: 8px;
            font-size: 11px; font-weight: 700; text-transform: uppercase;
            letter-spacing: 1.5px; color: var(--accent-blue); margin-bottom: 10px;
        }}
        .section-label .line {{ width: 20px; height: 1px; background: var(--accent-blue); }}
        .section-title {{ font-size: 26px; font-weight: 800; letter-spacing: -0.5px; color: var(--text-primary); }}
        .section-subtitle {{ font-size: 14px; color: var(--text-secondary); margin-top: 6px; }}

        /* ── Metric Cards ─────────────────────────────── */
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; }}
        .metric-card {{
            background: var(--bg-card); border: 1px solid var(--border-subtle);
            border-radius: var(--radius); padding: 24px 22px;
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative; overflow: hidden;
        }}
        .metric-card:hover {{
            background: var(--bg-card-hover); transform: translateY(-3px);
            border-color: var(--border-hover); box-shadow: var(--shadow-glow);
        }}
        .metric-card::before {{
            content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
            border-radius: var(--radius) var(--radius) 0 0;
        }}
        .metric-card.mc-blue::before {{ background: linear-gradient(90deg, var(--accent-blue), var(--accent-indigo)); }}
        .metric-card.mc-purple::before {{ background: linear-gradient(90deg, var(--accent-purple), var(--accent-pink)); }}
        .metric-card.mc-cyan::before {{ background: linear-gradient(90deg, var(--accent-cyan), var(--accent-emerald)); }}
        .metric-card.mc-emerald::before {{ background: linear-gradient(90deg, var(--accent-emerald), var(--accent-cyan)); }}
        .metric-card.mc-amber::before {{ background: linear-gradient(90deg, var(--accent-amber), var(--accent-pink)); }}
        .metric-card.mc-pink::before {{ background: linear-gradient(90deg, var(--accent-pink), var(--accent-purple)); }}
        .metric-card.mc-red::before {{ background: linear-gradient(90deg, var(--accent-red), var(--accent-amber)); }}
        .metric-label {{
            font-size: 11px; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.8px; color: var(--text-secondary); margin-bottom: 10px;
        }}
        .metric-value {{ font-size: 34px; font-weight: 900; letter-spacing: -1px; font-variant-numeric: tabular-nums; }}
        .metric-subtext {{ font-size: 11px; color: var(--text-muted); margin-top: 6px; }}

        /* ── Charts Grid ──────────────────────────────── */
        .charts-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
        @media (max-width: 900px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
        .chart-card {{
            background: var(--bg-card); border: 1px solid var(--border-subtle);
            border-radius: var(--radius); padding: 28px; position: relative;
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            min-height: 400px;
        }}
        .chart-card:hover {{ border-color: var(--border-hover); box-shadow: var(--shadow-glow); }}
        .chart-card h3 {{ font-size: 14px; font-weight: 700; margin-bottom: 4px; }}
        .chart-card .chart-desc {{ font-size: 12px; color: var(--text-muted); margin-bottom: 20px; }}
        .chart-card canvas {{ max-height: 320px; }}
        .chart-card.full-width {{ grid-column: 1 / -1; }}
        .chart-loader {{
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
            width: 32px; height: 32px; border: 3px solid rgba(79,143,255,0.15);
            border-top-color: var(--accent-blue); border-radius: 50%;
            animation: spin 0.8s linear infinite; z-index: 5;
        }}
        @keyframes spin {{ to {{ transform: translate(-50%, -50%) rotate(360deg); }} }}

        /* ── Config grid ──────────────────────────────── */
        .config-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }}
        .config-item {{
            background: var(--bg-card); border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm); padding: 18px 20px;
            display: flex; align-items: center; gap: 14px;
            transition: all 0.3s ease;
        }}
        .config-item:hover {{ background: var(--bg-card-hover); border-color: var(--border-hover); }}
        .config-icon {{
            width: 40px; height: 40px; border-radius: var(--radius-xs);
            display: flex; align-items: center; justify-content: center;
            font-size: 18px; flex-shrink: 0;
        }}
        .config-icon.ci-blue {{ background: rgba(79,143,255,0.1); }}
        .config-icon.ci-purple {{ background: rgba(168,85,247,0.1); }}
        .config-icon.ci-cyan {{ background: rgba(34,211,238,0.1); }}
        .config-icon.ci-emerald {{ background: rgba(52,211,153,0.1); }}
        .config-icon.ci-amber {{ background: rgba(251,191,36,0.1); }}
        .config-icon.ci-pink {{ background: rgba(244,114,182,0.1); }}
        .config-detail-label {{ font-size: 11px; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
        .config-detail-value {{ font-size: 15px; font-weight: 700; color: var(--text-primary); margin-top: 2px; }}

        /* ── History Table ─────────────────────────────── */
        .table-wrapper {{
            background: var(--bg-card); border: 1px solid var(--border-subtle);
            border-radius: var(--radius); overflow: hidden;
        }}
        .table-wrapper table {{ width: 100%; border-collapse: collapse; }}
        .table-wrapper thead th {{
            background: rgba(255, 255, 255, 0.02); padding: 14px 18px;
            font-size: 11px; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.8px; color: var(--text-secondary); text-align: left;
            border-bottom: 1px solid var(--border-subtle);
        }}
        .table-wrapper tbody td {{
            padding: 14px 18px; font-size: 13px; font-weight: 500;
            border-bottom: 1px solid var(--border-subtle);
            color: var(--text-primary); font-variant-numeric: tabular-nums;
        }}
        .table-wrapper tbody tr {{ cursor: pointer; transition: background 0.2s ease; }}
        .table-wrapper tbody tr:hover td {{ background: rgba(255, 255, 255, 0.03); }}
        .table-wrapper tbody tr:last-child td {{ border-bottom: none; }}
        .badge {{
            display: inline-flex; padding: 3px 10px; border-radius: 100px;
            font-size: 10px; font-weight: 700; letter-spacing: 0.3px; text-transform: uppercase;
        }}
        .badge--latest {{ background: rgba(52, 211, 153, 0.12); color: var(--accent-emerald); }}
        .expand-icon {{
            display: inline-flex; align-items: center; justify-content: center;
            width: 22px; height: 22px; border-radius: 6px;
            background: rgba(255,255,255,0.04); color: var(--text-muted);
            font-size: 12px; transition: all 0.3s ease; float: right;
        }}
        .expand-icon.open {{ transform: rotate(180deg); color: var(--accent-blue); }}
        .plot-row td {{ padding: 0 !important; border-bottom: 1px solid var(--border-subtle) !important; }}
        .history-charts-wrapper {{
            max-height: 0; overflow: hidden;
            transition: max-height 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }}
        .history-charts-wrapper.open {{ max-height: 600px; }}
        .history-charts-container {{
            padding: 24px; background: rgba(0,0,0,0.2);
        }}
        .history-charts-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px;
        }}
        .history-chart-item {{ text-align: center; }}
        .history-chart-item p {{ font-size: 12px; color: var(--text-secondary); margin-bottom: 10px; font-weight: 600; }}
        .history-chart-item canvas {{ max-height: 260px; }}

        /* ── Custom Dropdown ──────────────────────────── */
        .custom-select {{
            position: relative; width: 100%;
        }}
        .custom-select-trigger {{
            display: flex; align-items: center; justify-content: space-between;
            background: rgba(0,0,0,0.25); border: 1px solid var(--border-subtle);
            border-radius: var(--radius-xs); padding: 10px 14px;
            color: var(--text-primary); font-family: inherit; font-size: 14px;
            cursor: pointer; transition: border-color 0.2s, box-shadow 0.2s;
            user-select: none;
        }}
        .custom-select-trigger:hover {{ border-color: var(--border-hover); }}
        .custom-select-trigger.open {{
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 2px rgba(79,143,255,0.15);
        }}
        .custom-select-trigger .arrow {{
            width: 16px; height: 16px; display: flex; align-items: center;
            justify-content: center; transition: transform 0.3s ease;
            color: var(--text-muted); flex-shrink: 0;
        }}
        .custom-select-trigger.open .arrow {{ transform: rotate(180deg); }}
        .custom-select-options {{
            position: absolute; top: calc(100% + 4px); left: 0; right: 0;
            background: #14162d; border: 1px solid var(--border-hover);
            border-radius: var(--radius-xs); overflow-y: auto;
            z-index: 100; opacity: 0; visibility: hidden;
            transform: translateY(-10px) scaleY(0.95); transform-origin: top;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            box-shadow: 0 12px 40px rgba(0,0,0,0.4);
            max-height: 0;
        }}
        .custom-select-options.open {{
            opacity: 1; visibility: visible; transform: translateY(0) scaleY(1);
            max-height: 250px;
        }}
        .custom-select-options::-webkit-scrollbar {{ width: 4px; }}
        .custom-select-options::-webkit-scrollbar-track {{ background: transparent; }}
        .custom-select-options::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.1); border-radius: 4px; }}
        .custom-select-option {{
            padding: 9px 14px; font-size: 13px; cursor: pointer;
            transition: background 0.15s; color: var(--text-secondary);
        }}
        .custom-select-option:hover {{ background: rgba(79,143,255,0.1); color: var(--text-primary); }}
        .custom-select-option.selected {{ color: var(--accent-blue); font-weight: 600; background: rgba(79,143,255,0.06); }}
        .custom-select-search {{
            padding: 8px 14px; border-bottom: 1px solid var(--border-subtle);
            position: sticky; top: 0; background: #14162d; z-index: 1;
        }}
        .custom-select-search input {{
            width: 100%; background: rgba(0,0,0,0.3); border: 1px solid var(--border-subtle);
            border-radius: 4px; padding: 6px 10px; color: var(--text-primary);
            font-family: inherit; font-size: 12px; outline: none;
        }}
        .custom-select-search input::placeholder {{ color: var(--text-muted); }}

        /* ── Form & Inputs ────────────────────────────── */
        .form-card {{
            background: var(--gradient-card); border: 1px solid var(--border-subtle);
            border-radius: var(--radius); padding: 32px;
            backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
        }}
        .form-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 28px; }}
        .input-group {{ display: flex; flex-direction: column; gap: 8px; }}
        .input-group label {{
            font-size: 12px; font-weight: 700; color: var(--text-secondary);
            text-transform: uppercase; letter-spacing: 0.5px;
        }}
        .input-group input {{
            background: rgba(0,0,0,0.25); border: 1px solid var(--border-subtle);
            border-radius: var(--radius-xs); padding: 10px 14px;
            color: var(--text-primary); font-family: inherit; font-size: 14px;
            outline: none; transition: border-color 0.2s, box-shadow 0.2s;
        }}
        .input-group input::placeholder {{ color: var(--text-muted); }}
        .input-group input:focus {{
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 2px rgba(79,143,255,0.15);
        }}
        .btn {{
            background: var(--gradient-primary); color: white; border: none;
            padding: 14px 28px; border-radius: var(--radius-xs);
            font-weight: 700; font-size: 15px; cursor: pointer;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            width: 100%; position: relative; overflow: hidden;
            letter-spacing: 0.3px;
        }}
        .btn::before {{
            content: ''; position: absolute; inset: 0;
            background: linear-gradient(135deg, rgba(255,255,255,0.1), transparent);
            opacity: 0; transition: opacity 0.3s;
        }}
        .btn:hover {{ transform: translateY(-2px); box-shadow: 0 8px 30px rgba(79,143,255,0.3); }}
        .btn:hover::before {{ opacity: 1; }}
        .btn:active {{ transform: translateY(0); }}

        /* ── Result Box ───────────────────────────────── */
        .result-box {{
            margin-top: 28px; padding: 22px 24px; border-radius: var(--radius-sm);
            display: none; align-items: center; gap: 16px;
            border: 1px solid; animation: fadeSlideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
        }}
        .result-box.serious {{ background: rgba(248, 113, 113, 0.08); border-color: rgba(248, 113, 113, 0.2); }}
        .result-box.safe {{ background: rgba(52, 211, 153, 0.08); border-color: rgba(52, 211, 153, 0.2); }}
        .result-box .res-icon {{ font-size: 36px; flex-shrink: 0; }}
        .result-box .res-title {{ font-size: 18px; font-weight: 800; margin-bottom: 4px; }}
        .result-box .res-desc {{ font-size: 13px; opacity: 0.85; color: var(--text-secondary); }}
        .result-box.serious .res-title {{ color: var(--accent-red); }}
        .result-box.safe .res-title {{ color: var(--accent-emerald); }}

        /* ── Footer ───────────────────────────────────── */
        .footer {{
            border-top: 1px solid var(--border-subtle); padding: 32px 0;
            text-align: center; font-size: 13px; color: var(--text-muted);
        }}
        .footer a {{ color: var(--accent-blue); text-decoration: none; }}
        .footer a:hover {{ text-decoration: underline; }}

        /* ── Animations ───────────────────────────────── */
        @keyframes fadeSlideUp {{
            from {{ opacity: 0; transform: translateY(14px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes shimmer {{
            0% {{ background-position: -200% 0; }}
            100% {{ background-position: 200% 0; }}
        }}
        .metric-value .shimmer {{
            background: linear-gradient(90deg, var(--text-primary) 25%, var(--accent-blue) 50%, var(--text-primary) 75%);
            background-size: 200% 100%;
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: shimmer 3s ease infinite;
        }}

        /* ── Responsive ───────────────────────────────── */
        @media (max-width: 768px) {{
            .header h1 {{ font-size: 32px; }}
            .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .charts-grid {{ grid-template-columns: 1fr; }}
            .config-grid {{ grid-template-columns: 1fr; }}
            .form-grid {{ grid-template-columns: 1fr; }}
            .container {{ padding: 0 16px; }}
            .section {{ padding: 36px 0; }}
        }}
    </style>
</head>
<body>

    <!-- ═══════════════════════════════════════════════════════ -->
    <!-- HEADER                                                  -->
    <!-- ═══════════════════════════════════════════════════════ -->
    <header class="header">
        <div class="container">
            <div class="header-badge"><span class="dot"></span> Model Serving</div>
            <h1>PharmaGuard</h1>
            <p>Real-time Adverse Drug Reaction severity predictions powered by <strong>XGBoost</strong>, trained on <strong>FDA FAERS</strong> data. Serving model <strong>{model_version}</strong>.</p>
        </div>
    </header>

    <!-- ═══════════════════════════════════════════════════════ -->
    <!-- CURRENT MODEL METRICS                                   -->
    <!-- ═══════════════════════════════════════════════════════ -->
    <section class="section reveal" id="metrics-section">
        <div class="container">
            <div class="section-header">
                <div class="section-label"><span class="line"></span> Current Model</div>
                <h2 class="section-title">Performance Metrics</h2>
                <p class="section-subtitle">Evaluation on held-out test set · Model {model_version} · Threshold {threshold}</p>
            </div>
            <div class="metrics-grid">
                <div class="metric-card mc-blue">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value" style="color: var(--accent-blue)"><span class="counter" data-target="{precision*100:.1f}">0</span>%</div>
                    <div class="metric-subtext">Positive predictive value</div>
                </div>
                <div class="metric-card mc-purple">
                    <div class="metric-label">Recall</div>
                    <div class="metric-value" style="color: var(--accent-purple)"><span class="counter" data-target="{recall*100:.1f}">0</span>%</div>
                    <div class="metric-subtext">Sensitivity / TPR</div>
                </div>
                <div class="metric-card mc-cyan">
                    <div class="metric-label">F1 Score</div>
                    <div class="metric-value" style="color: var(--accent-cyan)"><span class="counter" data-target="{f1*100:.1f}">0</span>%</div>
                    <div class="metric-subtext">Harmonic mean of P & R</div>
                </div>
                <div class="metric-card mc-emerald">
                    <div class="metric-label">ROC-AUC</div>
                    <div class="metric-value" style="color: var(--accent-emerald)"><span class="counter" data-target="{roc_auc*100:.1f}">0</span>%</div>
                    <div class="metric-subtext">Area under ROC curve</div>
                </div>
                <div class="metric-card mc-pink">
                    <div class="metric-label">PR-AUC</div>
                    <div class="metric-value" style="color: var(--accent-pink)"><span class="counter" data-target="{pr_auc*100:.1f}">0</span>%</div>
                    <div class="metric-subtext">Area under PR curve</div>
                </div>
                <div class="metric-card mc-amber">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value" style="color: var(--accent-amber)"><span class="counter" data-target="{accuracy*100:.1f}">0</span>%</div>
                    <div class="metric-subtext">Overall correctness</div>
                </div>
                <div class="metric-card mc-red">
                    <div class="metric-label">Specificity</div>
                    <div class="metric-value" style="color: var(--accent-red)"><span class="counter" data-target="{specificity*100:.1f}">0</span>%</div>
                    <div class="metric-subtext">True negative rate</div>
                </div>
            </div>
        </div>
    </section>

    <!-- ═══════════════════════════════════════════════════════ -->
    <!-- INTERACTIVE CHARTS                                      -->
    <!-- ═══════════════════════════════════════════════════════ -->
    <section class="section reveal" id="charts-section">
        <div class="container">
            <div class="section-header">
                <div class="section-label"><span class="line"></span> Analytics</div>
                <h2 class="section-title">Model Analytics</h2>
                <p class="section-subtitle">Interactive visualizations of the deployed model's performance</p>
            </div>
            <div class="charts-grid">
                <div class="chart-card">
                    <div class="chart-loader" id="loader-confusion"></div>
                    <h3>Confusion Matrix</h3>
                    <p class="chart-desc">Test set predictions breakdown</p>
                    <canvas id="chart-confusion"></canvas>
                </div>
                <div class="chart-card">
                    <div class="chart-loader" id="loader-radar"></div>
                    <h3>Train / Val / Test Comparison</h3>
                    <p class="chart-desc">Radar chart across all evaluation metrics</p>
                    <canvas id="chart-radar"></canvas>
                </div>
                <div class="chart-card">
                    <div class="chart-loader" id="loader-bar"></div>
                    <h3>Test Metrics Breakdown</h3>
                    <p class="chart-desc">Bar chart of all test set metrics</p>
                    <canvas id="chart-bar"></canvas>
                </div>
                <div class="chart-card">
                    <div class="chart-loader" id="loader-history"></div>
                    <h3>Historical Performance</h3>
                    <p class="chart-desc">Metric trends across training runs</p>
                    <canvas id="chart-history"></canvas>
                </div>
            </div>
        </div>
    </section>

    <!-- ═══════════════════════════════════════════════════════ -->
    <!-- MODEL CONFIGURATION                                     -->
    <!-- ═══════════════════════════════════════════════════════ -->
    <section class="section reveal" id="config-section">
        <div class="container">
            <div class="section-header">
                <div class="section-label"><span class="line"></span> Configuration</div>
                <h2 class="section-title">Model Configuration</h2>
                <p class="section-subtitle">Hyperparameters and training configuration for the deployed model</p>
            </div>
            <div class="config-grid">
                <div class="config-item">
                    <div class="config-icon ci-blue">🎯</div>
                    <div><div class="config-detail-label">Decision Threshold</div><div class="config-detail-value">{threshold}</div></div>
                </div>
                <div class="config-item">
                    <div class="config-icon ci-purple">🧠</div>
                    <div><div class="config-detail-label">Threshold Strategy</div><div class="config-detail-value">{threshold_strategy}</div></div>
                </div>
                <div class="config-item">
                    <div class="config-icon ci-cyan">🌲</div>
                    <div><div class="config-detail-label">N Estimators</div><div class="config-detail-value">{n_estimators}</div></div>
                </div>
                <div class="config-item">
                    <div class="config-icon ci-emerald">📏</div>
                    <div><div class="config-detail-label">Max Depth</div><div class="config-detail-value">{max_depth}</div></div>
                </div>
                <div class="config-item">
                    <div class="config-icon ci-amber">⚡</div>
                    <div><div class="config-detail-label">Learning Rate</div><div class="config-detail-value">{learning_rate}</div></div>
                </div>
                <div class="config-item">
                    <div class="config-icon ci-pink">⚖️</div>
                    <div><div class="config-detail-label">Scale Pos Weight</div><div class="config-detail-value">{scale_pos_weight}</div></div>
                </div>
                <div class="config-item">
                    <div class="config-icon ci-blue">📊</div>
                    <div><div class="config-detail-label">Eval Metric</div><div class="config-detail-value">{eval_metric}</div></div>
                </div>
                <div class="config-item">
                    <div class="config-icon ci-purple">⏱️</div>
                    <div><div class="config-detail-label">Early Stopping</div><div class="config-detail-value">{early_stopping} rounds</div></div>
                </div>
                <div class="config-item">
                    <div class="config-icon ci-cyan">📅</div>
                    <div><div class="config-detail-label">Training Quarters</div><div class="config-detail-value">{quarters}</div></div>
                </div>
            </div>
        </div>
    </section>

    <!-- ═══════════════════════════════════════════════════════ -->
    <!-- TRAINING HISTORY                                        -->
    <!-- ═══════════════════════════════════════════════════════ -->
    <section class="section reveal" id="history-section">
        <div class="container">
            <div class="section-header">
                <div class="section-label"><span class="line"></span> History</div>
                <h2 class="section-title">Training History</h2>
                <p class="section-subtitle">Click any run to expand dynamic evaluation charts</p>
            </div>
            {_render_empty_or_table(history, history_rows)}
        </div>
    </section>

    <!-- ═══════════════════════════════════════════════════════ -->
    <!-- LIVE PREDICTION                                         -->
    <!-- ═══════════════════════════════════════════════════════ -->
    <section class="section reveal" id="predict-section">
        <div class="container">
            <div class="section-header">
                <div class="section-label"><span class="line"></span> Try It</div>
                <h2 class="section-title">Live Prediction</h2>
                <p class="section-subtitle">Test the <code style="background:rgba(79,143,255,0.1);padding:2px 6px;border-radius:4px;font-size:12px;color:var(--accent-blue)">/predict</code> endpoint directly from the browser</p>
            </div>
            <div class="form-card">
                <form id="predict-form" onsubmit="event.preventDefault(); makePrediction();">
                    <div class="form-grid">
                        <div class="input-group">
                            <label>Age Group</label>
                            <div class="custom-select" data-name="f_age" data-value="18-44">
                                <div class="custom-select-trigger" onclick="toggleDropdown(this)">
                                    <span class="selected-text">18-44</span>
                                    <span class="arrow"><svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg></span>
                                </div>
                                <div class="custom-select-options">
                                    <div class="custom-select-option selected" data-value="0-17">0-17</div>
                                    <div class="custom-select-option" data-value="18-44">18-44</div>
                                    <div class="custom-select-option" data-value="45-64">45-64</div>
                                    <div class="custom-select-option" data-value="65-74">65-74</div>
                                    <div class="custom-select-option" data-value="75+">75+</div>
                                    <div class="custom-select-option" data-value="unknown">Unknown</div>
                                </div>
                            </div>
                        </div>
                        <div class="input-group">
                            <label>Sex</label>
                            <div class="custom-select" data-name="f_sex" data-value="M">
                                <div class="custom-select-trigger" onclick="toggleDropdown(this)">
                                    <span class="selected-text">Male (M)</span>
                                    <span class="arrow"><svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg></span>
                                </div>
                                <div class="custom-select-options">
                                    <div class="custom-select-option selected" data-value="M">Male (M)</div>
                                    <div class="custom-select-option" data-value="F">Female (F)</div>
                                    <div class="custom-select-option" data-value="UNK">Unknown (UNK)</div>
                                </div>
                            </div>
                        </div>
                        <div class="input-group">
                            <label>Route</label>
                            <div class="custom-select" data-name="f_route" data-value="Oral">
                                <div class="custom-select-trigger" onclick="toggleDropdown(this)">
                                    <span class="selected-text">Oral</span>
                                    <span class="arrow"><svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg></span>
                                </div>
                                <div class="custom-select-options">
                                    <div class="custom-select-search"><input type="text" placeholder="Search routes..." oninput="filterOptions(this)"></div>
                                    <div class="custom-select-option selected" data-value="Oral">Oral</div>
                                    <div class="custom-select-option" data-value="Oral use">Oral use</div>
                                    <div class="custom-select-option" data-value="Subcutaneous">Subcutaneous</div>
                                    <div class="custom-select-option" data-value="Subcutaneous use">Subcutaneous use</div>
                                    <div class="custom-select-option" data-value="Topical">Topical</div>
                                    <div class="custom-select-option" data-value="Intravenous (not otherwise specified)">Intravenous (not otherwise specified)</div>
                                    <div class="custom-select-option" data-value="Intravenous use">Intravenous use</div>
                                    <div class="custom-select-option" data-value="Intravenous drip">Intravenous drip</div>
                                    <div class="custom-select-option" data-value="Intravenous bolus">Intravenous bolus</div>
                                    <div class="custom-select-option" data-value="Intramuscular">Intramuscular</div>
                                    <div class="custom-select-option" data-value="Intramuscular use">Intramuscular use</div>
                                    <div class="custom-select-option" data-value="Ophthalmic">Ophthalmic</div>
                                    <div class="custom-select-option" data-value="Respiratory (inhalation)">Respiratory (inhalation)</div>
                                    <div class="custom-select-option" data-value="Inhalation use">Inhalation use</div>
                                    <div class="custom-select-option" data-value="Nasal">Nasal</div>
                                    <div class="custom-select-option" data-value="Transdermal">Transdermal</div>
                                    <div class="custom-select-option" data-value="Transdermal use">Transdermal use</div>
                                    <div class="custom-select-option" data-value="Transplacental">Transplacental</div>
                                    <div class="custom-select-option" data-value="Intra-uterine">Intra-uterine</div>
                                    <div class="custom-select-option" data-value="Cutaneous">Cutaneous</div>
                                    <div class="custom-select-option" data-value="Intrathecal">Intrathecal</div>
                                    <div class="custom-select-option" data-value="Intraperitoneal">Intraperitoneal</div>
                                    <div class="custom-select-option" data-value="Intraperitoneal use">Intraperitoneal use</div>
                                    <div class="custom-select-option" data-value="Sublingual">Sublingual</div>
                                    <div class="custom-select-option" data-value="Vaginal">Vaginal</div>
                                    <div class="custom-select-option" data-value="Rectal">Rectal</div>
                                    <div class="custom-select-option" data-value="Rectal use">Rectal use</div>
                                    <div class="custom-select-option" data-value="Buccal">Buccal</div>
                                    <div class="custom-select-option" data-value="Parenteral">Parenteral</div>
                                    <div class="custom-select-option" data-value="Epidural">Epidural</div>
                                    <div class="custom-select-option" data-value="Intravitreal">Intravitreal</div>
                                    <div class="custom-select-option" data-value="Intravesical">Intravesical</div>
                                    <div class="custom-select-option" data-value="Intraocular">Intraocular</div>
                                    <div class="custom-select-option" data-value="Intra-articular">Intra-articular</div>
                                    <div class="custom-select-option" data-value="Intra-arterial">Intra-arterial</div>
                                    <div class="custom-select-option" data-value="Intracardiac">Intracardiac</div>
                                    <div class="custom-select-option" data-value="Subdermal">Subdermal</div>
                                    <div class="custom-select-option" data-value="Other">Other</div>
                                    <div class="custom-select-option" data-value="Unknown">Unknown</div>
                                </div>
                            </div>
                        </div>
                        <div class="input-group">
                            <label>Drug Name</label>
                            <input type="text" id="f_drug" placeholder="e.g. ASPIRIN" value="ASPIRIN" required>
                        </div>
                        <div class="input-group">
                            <label>Polypharmacy Count</label>
                            <input type="number" id="f_poly" placeholder="Number of concurrent drugs" value="3" min="0" required>
                        </div>
                    </div>
                    <button type="submit" class="btn" id="btn-predict">
                        Predict Severity
                    </button>
                </form>

                <div id="result-box" class="result-box">
                    <div class="res-icon" id="res-icon"></div>
                    <div>
                        <div class="res-title" id="res-title"></div>
                        <div class="res-desc" id="res-desc"></div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- ═══════════════════════════════════════════════════════ -->
    <!-- FOOTER                                                  -->
    <!-- ═══════════════════════════════════════════════════════ -->
    <footer class="footer">
        <div class="container">
            <p>PharmaGuard · Built with FastAPI + XGBoost · <a href="/docs">Swagger Docs</a> · <a href="/redoc">ReDoc</a> · <a href="https://github.com/pchhabra07/PharmaGuard" target="_blank">GitHub</a></p>
        </div>
    </footer>

    <!-- ═══════════════════════════════════════════════════════ -->
    <!-- JAVASCRIPT                                              -->
    <!-- ═══════════════════════════════════════════════════════ -->
    <script>
    const baseUrl = "{base_url}";

    // ── Scroll Reveal ─────────────────────────────────
    const observer = new IntersectionObserver((entries) => {{
        entries.forEach(entry => {{
            if (entry.isIntersecting) {{
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }}
        }});
    }}, {{ threshold: 0.08 }});
    document.querySelectorAll('.reveal').forEach(el => observer.observe(el));

    // ── Animated Counters ─────────────────────────────
    function animateCounters() {{
        document.querySelectorAll('.counter').forEach(counter => {{
            const target = parseFloat(counter.dataset.target);
            const duration = 1400;
            const start = performance.now();
            function update(now) {{
                const elapsed = now - start;
                const progress = Math.min(elapsed / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 4);
                counter.textContent = (target * eased).toFixed(1);
                if (progress < 1) requestAnimationFrame(update);
                else counter.textContent = target.toFixed(1);
            }}
            requestAnimationFrame(update);
        }});
    }}
    // Trigger counters when metrics section enters view
    const metricsObs = new IntersectionObserver((entries) => {{
        entries.forEach(entry => {{
            if (entry.isIntersecting) {{
                animateCounters();
                metricsObs.unobserve(entry.target);
            }}
        }});
    }}, {{ threshold: 0.3 }});
    const metricsSection = document.getElementById('metrics-section');
    if (metricsSection) metricsObs.observe(metricsSection);

    // ── Custom Dropdown ───────────────────────────────
    function toggleDropdown(trigger) {{
        const select = trigger.closest('.custom-select');
        const options = select.querySelector('.custom-select-options');
        const isOpen = options.classList.contains('open');

        // Close all dropdowns first
        closeAllDropdowns();

        if (!isOpen) {{
            trigger.classList.add('open');
            options.classList.add('open');
            const searchInput = options.querySelector('.custom-select-search input');
            if (searchInput) setTimeout(() => searchInput.focus(), 50);
        }}
    }}

    function closeAllDropdowns() {{
        document.querySelectorAll('.custom-select-options.open').forEach(opt => {{
            opt.classList.remove('open');
            opt.closest('.custom-select').querySelector('.custom-select-trigger').classList.remove('open');
        }});
    }}

    document.addEventListener('click', (e) => {{
        if (!e.target.closest('.custom-select')) closeAllDropdowns();
    }});

    document.querySelectorAll('.custom-select-option').forEach(option => {{
        option.addEventListener('click', function() {{
            const select = this.closest('.custom-select');
            const trigger = select.querySelector('.custom-select-trigger');
            const options = select.querySelector('.custom-select-options');

            // Update selection
            options.querySelectorAll('.custom-select-option').forEach(o => o.classList.remove('selected'));
            this.classList.add('selected');
            select.dataset.value = this.dataset.value;
            trigger.querySelector('.selected-text').textContent = this.textContent;

            // Close
            trigger.classList.remove('open');
            options.classList.remove('open');
        }});
    }});

    function filterOptions(input) {{
        const query = input.value.toLowerCase();
        const options = input.closest('.custom-select-options').querySelectorAll('.custom-select-option');
        options.forEach(opt => {{
            opt.style.display = opt.textContent.toLowerCase().includes(query) ? '' : 'none';
        }});
    }}

    // ── Charts ────────────────────────────────────────
    let chartDataCache = null;

    // Fetch data immediately
    async function fetchChartData() {{
        try {{
            const res = await fetch(baseUrl + '/model/charts');
            chartDataCache = await res.json();
        }} catch (e) {{
            console.error('Failed to load chart data:', e);
            document.querySelectorAll('.chart-loader').forEach(el => el.style.display = 'none');
        }}
    }}

    const chartsObs = new IntersectionObserver((entries) => {{
        entries.forEach(entry => {{
            if (entry.isIntersecting) {{
                renderAllCharts();
                chartsObs.unobserve(entry.target);
            }}
        }});
    }}, {{ threshold: 0.1 }});

    async function renderAllCharts() {{
        // Wait for fetch if triggered immediately
        while (!chartDataCache) {{
            await new Promise(r => setTimeout(r, 100));
        }}

        const data = chartDataCache;
        if (!data.latest) return;

        Chart.defaults.color = '#8892a8';
        Chart.defaults.borderColor = 'rgba(255,255,255,0.04)';
        Chart.defaults.font.family = "'Inter', sans-serif";

        document.getElementById('loader-confusion').style.display = 'none';
        renderConfusionMatrix(data.latest.confusion_matrix);
        
        document.getElementById('loader-radar').style.display = 'none';
        renderRadar(data.latest.splits_comparison);
        
        document.getElementById('loader-bar').style.display = 'none';
        renderBarChart(data.latest.test_metrics);
        
        document.getElementById('loader-history').style.display = 'none';
        renderHistoryChart(data.history_trend);
    }}

    function renderConfusionMatrix(cm) {{
        const ctx = document.getElementById('chart-confusion');
        if (!ctx) return;

        const total = cm.tn + cm.fp + cm.fn + cm.tp;
        
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: ['True Neg (TN)', 'False Pos (FP)', 'False Neg (FN)', 'True Pos (TP)'],
                datasets: [{{
                    data: [cm.tn, cm.fp, cm.fn, cm.tp],
                    backgroundColor: [
                        'rgba(52, 211, 153, 0.7)',
                        'rgba(248, 113, 113, 0.5)',
                        'rgba(251, 191, 36, 0.5)',
                        'rgba(79, 143, 255, 0.7)',
                    ],
                    borderColor: [
                        'rgba(52, 211, 153, 1)',
                        'rgba(248, 113, 113, 1)',
                        'rgba(251, 191, 36, 1)',
                        'rgba(79, 143, 255, 1)',
                    ],
                    borderWidth: 1,
                    borderRadius: 6,
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: true,
                animation: {{ duration: 1200, easing: 'easeOutQuart' }},
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: (ctx) => {{
                                const val = ctx.raw;
                                const pct = ((val / total) * 100).toFixed(1);
                                return ` ${{val.toLocaleString()}} (${{pct}}%)`;
                            }}
                        }},
                        backgroundColor: 'rgba(15, 17, 41, 0.95)',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1,
                        cornerRadius: 8,
                        padding: 12,
                    }}
                }},
                scales: {{
                    x: {{
                        ticks: {{ callback: v => v >= 1000 ? (v/1000).toFixed(0) + 'k' : v }},
                        grid: {{ color: 'rgba(255,255,255,0.03)' }}
                    }},
                    y: {{
                        grid: {{ display: false }}
                    }}
                }}
            }}
        }});
    }}

    function renderRadar(splits) {{
        const ctx = document.getElementById('chart-radar');
        if (!ctx) return;

        new Chart(ctx, {{
            type: 'radar',
            data: {{
                labels: splits.labels,
                datasets: [
                    {{
                        label: 'Train',
                        data: splits.train,
                        borderColor: '#34d399',
                        backgroundColor: 'rgba(52, 211, 153, 0.08)',
                        pointBackgroundColor: '#34d399',
                        borderWidth: 2,
                        pointRadius: 3,
                    }},
                    {{
                        label: 'Validation',
                        data: splits.val,
                        borderColor: '#fbbf24',
                        backgroundColor: 'rgba(251, 191, 36, 0.06)',
                        pointBackgroundColor: '#fbbf24',
                        borderWidth: 2,
                        pointRadius: 3,
                    }},
                    {{
                        label: 'Test',
                        data: splits.test,
                        borderColor: '#4f8fff',
                        backgroundColor: 'rgba(79, 143, 255, 0.08)',
                        pointBackgroundColor: '#4f8fff',
                        borderWidth: 2,
                        pointRadius: 3,
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                animation: {{ duration: 1200, easing: 'easeOutQuart' }},
                scales: {{
                    r: {{
                        beginAtZero: true, max: 1,
                        ticks: {{ stepSize: 0.2, font: {{ size: 10 }}, backdropColor: 'transparent' }},
                        grid: {{ color: 'rgba(255,255,255,0.04)' }},
                        angleLines: {{ color: 'rgba(255,255,255,0.06)' }},
                        pointLabels: {{ font: {{ size: 11, weight: '600' }}, color: '#8892a8' }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        position: 'bottom',
                        labels: {{ padding: 20, usePointStyle: true, pointStyle: 'circle', font: {{ size: 12, weight: '500' }} }}
                    }},
                    tooltip: {{
                        callbacks: {{ label: ctx => ` ${{ctx.dataset.label}}: ${{(ctx.raw * 100).toFixed(2)}}%` }},
                        backgroundColor: 'rgba(15, 17, 41, 0.95)',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1, cornerRadius: 8, padding: 12,
                    }}
                }}
            }}
        }});
    }}

    function renderBarChart(metrics) {{
        const ctx = document.getElementById('chart-bar');
        if (!ctx) return;

        const labels = Object.keys(metrics).map(k => k.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase()));
        const values = Object.values(metrics);
        const colors = [
            '#4f8fff', '#a855f7', '#22d3ee',
            '#34d399', '#f87171', '#fbbf24', '#f472b6'
        ];

        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: labels,
                datasets: [{{
                    data: values.map(v => v * 100),
                    backgroundColor: colors.map(c => c + '33'),
                    borderColor: colors,
                    borderWidth: 1.5,
                    borderRadius: 8,
                    borderSkipped: false,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                animation: {{ duration: 1200, easing: 'easeOutQuart' }},
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{ label: ctx => ` ${{ctx.raw.toFixed(2)}}%` }},
                        backgroundColor: 'rgba(15, 17, 41, 0.95)',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1, cornerRadius: 8, padding: 12,
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true, max: 100,
                        ticks: {{ callback: v => v + '%' }},
                        grid: {{ color: 'rgba(255,255,255,0.03)' }}
                    }},
                    x: {{
                        ticks: {{ font: {{ size: 11, weight: '600' }} }},
                        grid: {{ display: false }}
                    }}
                }}
            }}
        }});
    }}

    function renderHistoryChart(trend) {{
        const ctx = document.getElementById('chart-history');
        if (!ctx) return;

        const labels = trend.map(t => t.version);

        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [
                    {{
                        label: 'F1 Score',
                        data: trend.map(t => t.f1 * 100),
                        borderColor: '#22d3ee',
                        backgroundColor: 'rgba(34, 211, 238, 0.1)',
                        tension: 0.4, fill: true, borderWidth: 2.5,
                        pointBackgroundColor: '#22d3ee',
                        pointBorderColor: '#06070d', pointBorderWidth: 2, pointRadius: 5,
                    }},
                    {{
                        label: 'ROC-AUC',
                        data: trend.map(t => t.roc_auc * 100),
                        borderColor: '#34d399',
                        backgroundColor: 'rgba(52, 211, 153, 0.05)',
                        tension: 0.4, fill: true, borderWidth: 2.5,
                        pointBackgroundColor: '#34d399',
                        pointBorderColor: '#06070d', pointBorderWidth: 2, pointRadius: 5,
                    }},
                    {{
                        label: 'PR-AUC',
                        data: trend.map(t => t.pr_auc * 100),
                        borderColor: '#f472b6',
                        backgroundColor: 'rgba(244, 114, 182, 0.05)',
                        tension: 0.4, fill: true, borderWidth: 2.5,
                        pointBackgroundColor: '#f472b6',
                        pointBorderColor: '#06070d', pointBorderWidth: 2, pointRadius: 5,
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                animation: {{ duration: 1200, easing: 'easeOutQuart' }},
                plugins: {{
                    legend: {{
                        position: 'bottom',
                        labels: {{ padding: 20, usePointStyle: true, pointStyle: 'circle', font: {{ size: 12, weight: '500' }} }}
                    }},
                    tooltip: {{
                        mode: 'index', intersect: false,
                        callbacks: {{ label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.raw.toFixed(2)}}%` }},
                        backgroundColor: 'rgba(15, 17, 41, 0.95)',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1, cornerRadius: 8, padding: 12,
                    }}
                }},
                scales: {{
                    y: {{
                        ticks: {{ callback: v => v + '%' }},
                        grid: {{ color: 'rgba(255,255,255,0.03)' }}
                    }},
                    x: {{
                        ticks: {{ font: {{ size: 12, weight: '600' }} }},
                        grid: {{ display: false }}
                    }}
                }}
            }}
        }});
    }}

    // ── Per-run charts (history table expansion) ──────
    function togglePlots(rowId, version) {{
        const wrapper = document.getElementById('wrapper-' + rowId);
        const expandIcon = document.getElementById('expand-' + rowId);

        if (wrapper.classList.contains('open')) {{
            wrapper.classList.remove('open');
            if (expandIcon) expandIcon.classList.remove('open');
        }} else {{
            // Hide all others
            document.querySelectorAll('.history-charts-wrapper').forEach(w => {{ w.classList.remove('open'); }});
            document.querySelectorAll('.expand-icon').forEach(e => {{ e.classList.remove('open'); }});

            wrapper.classList.add('open');
            if (expandIcon) expandIcon.classList.add('open');

            // Render charts if not already done
            const container = document.getElementById('plot-container-' + rowId);
            if (!container.dataset.rendered) {{
                container.innerHTML = '<div class="chart-loader" style="position:relative; margin: 40px auto; transform:none; left:auto; top:auto;"></div>';
                renderRunCharts(container, rowId);
                container.dataset.rendered = 'true';
            }}
        }}
    }}

    async function renderRunCharts(container, rowId) {{
        try {{
            const res = await fetch(baseUrl + '/model/history');
            const data = await res.json();
            const runs = data.runs;
            const runIdx = runs.length - 1 - rowId;
            if (runIdx < 0 || runIdx >= runs.length) return;
            const run = runs[runIdx];

            const test_m = run.metrics?.test || {{}};
            const cm = test_m.confusion_matrix || [[0,0],[0,0]];

            // Build container HTML
            container.innerHTML = `
                <div class="history-charts-grid">
                    <div class="history-chart-item">
                        <p>Confusion Matrix</p>
                        <canvas id="hist-cm-${{rowId}}"></canvas>
                    </div>
                    <div class="history-chart-item">
                        <p>Test Metrics</p>
                        <canvas id="hist-bar-${{rowId}}"></canvas>
                    </div>
                </div>
            `;

            // Confusion matrix
            const total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1];
            new Chart(document.getElementById('hist-cm-' + rowId), {{
                type: 'bar',
                data: {{
                    labels: ['True Neg', 'False Pos', 'False Neg', 'True Pos'],
                    datasets: [{{
                        data: [cm[0][0], cm[0][1], cm[1][0], cm[1][1]],
                        backgroundColor: [
                            'rgba(52,211,153,0.6)', 'rgba(248,113,113,0.4)',
                            'rgba(251,191,36,0.4)', 'rgba(79,143,255,0.6)'
                        ],
                        borderColor: [
                            'rgba(52,211,153,1)', 'rgba(248,113,113,1)',
                            'rgba(251,191,36,1)', 'rgba(79,143,255,1)'
                        ],
                        borderWidth: 1, borderRadius: 4
                    }}]
                }},
                options: {{
                    indexAxis: 'y', responsive: true, maintainAspectRatio: true,
                    animation: {{ duration: 800, easing: 'easeOutQuart' }},
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            callbacks: {{ label: ctx => ` ${{ctx.raw.toLocaleString()}} (${{((ctx.raw/total)*100).toFixed(1)}}%)` }},
                            backgroundColor: 'rgba(15,17,41,0.95)', borderColor: 'rgba(255,255,255,0.1)',
                            borderWidth: 1, cornerRadius: 8, padding: 10,
                        }}
                    }},
                    scales: {{
                        x: {{ ticks: {{ callback: v => v >= 1000 ? (v/1000).toFixed(0) + 'k' : v }}, grid: {{ color: 'rgba(255,255,255,0.03)' }} }},
                        y: {{ grid: {{ display: false }} }}
                    }}
                }}
            }});

            // Metrics bar
            const metricKeys = ['precision', 'recall', 'f1', 'accuracy', 'roc_auc', 'pr_auc'];
            const metricLabels = metricKeys.map(k => k.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase()));
            const metricVals = metricKeys.map(k => (test_m[k] || 0) * 100);
            const barColors = ['#4f8fff', '#a855f7', '#22d3ee', '#34d399', '#fbbf24', '#f472b6'];

            new Chart(document.getElementById('hist-bar-' + rowId), {{
                type: 'bar',
                data: {{
                    labels: metricLabels,
                    datasets: [{{
                        data: metricVals,
                        backgroundColor: barColors.map(c => c + '33'),
                        borderColor: barColors,
                        borderWidth: 1.5, borderRadius: 6, borderSkipped: false,
                    }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: true,
                    animation: {{ duration: 800, easing: 'easeOutQuart' }},
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            callbacks: {{ label: ctx => ` ${{ctx.raw.toFixed(2)}}%` }},
                            backgroundColor: 'rgba(15,17,41,0.95)', borderColor: 'rgba(255,255,255,0.1)',
                            borderWidth: 1, cornerRadius: 8, padding: 10,
                        }}
                    }},
                    scales: {{
                        y: {{ beginAtZero: true, max: 100, ticks: {{ callback: v => v + '%' }}, grid: {{ color: 'rgba(255,255,255,0.03)' }} }},
                        x: {{ ticks: {{ font: {{ size: 10, weight: '600' }} }}, grid: {{ display: false }} }}
                    }}
                }}
            }});

        }} catch (e) {{
            container.innerHTML = '<p style="color:var(--text-muted);padding:20px;text-align:center;">Could not load run data.</p>';
            console.error(e);
        }}
    }}

    // ── Prediction ────────────────────────────────────
    async function makePrediction() {{
        const btn = document.getElementById('btn-predict');
        const resBox = document.getElementById('result-box');

        btn.textContent = 'Analyzing...';
        btn.style.opacity = '0.7';

        const payload = {{
            age_group: document.querySelector('[data-name="f_age"]').dataset.value,
            sex: document.querySelector('[data-name="f_sex"]').dataset.value,
            route: document.querySelector('[data-name="f_route"]').dataset.value,
            drug_name_normalized: document.getElementById('f_drug').value,
            polypharmacy_count: parseInt(document.getElementById('f_poly').value)
        }};

        try {{
            const res = await fetch(baseUrl + '/predict', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(payload)
            }});
            const data = await res.json();

            resBox.style.display = 'flex';
            if (data.is_serious) {{
                resBox.className = 'result-box serious';
                document.getElementById('res-icon').innerHTML = '⚠️';
                document.getElementById('res-title').innerText = 'High Risk — Serious ADR Predicted';
                document.getElementById('res-desc').innerText = `Probability: ${{(data.probability * 100).toFixed(2)}}% · Threshold: ${{data.threshold}} · Model: ${{data.model_version}}`;
            }} else {{
                resBox.className = 'result-box safe';
                document.getElementById('res-icon').innerHTML = '✅';
                document.getElementById('res-title').innerText = 'Low Risk — Not Serious';
                document.getElementById('res-desc').innerText = `Probability: ${{(data.probability * 100).toFixed(2)}}% · Threshold: ${{data.threshold}} · Model: ${{data.model_version}}`;
            }}

        }} catch (e) {{
            resBox.style.display = 'flex';
            resBox.className = 'result-box serious';
            document.getElementById('res-icon').innerHTML = '❌';
            document.getElementById('res-title').innerText = 'Prediction Failed';
            document.getElementById('res-desc').innerText = 'Check browser console for details.';
            console.error(e);
        }} finally {{
            btn.textContent = 'Predict Severity';
            btn.style.opacity = '1';
        }}
    }}

    // ── Init ──────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', () => {{
        fetchChartData();
        const chartsSection = document.getElementById('charts-section');
        if (chartsSection) chartsObs.observe(chartsSection);
    }});
    </script>
</body>
</html>"""


def _build_history_rows(history: list) -> str:
    if not history:
        return ""
    rows = []
    for i, run in enumerate(reversed(history)):
        test_m = run.get("metrics", {}).get("test", {})
        is_latest = i == 0
        badge = ' <span class="badge badge--latest">latest</span>' if is_latest else ""
        timestamp = run.get("timestamp", "N/A").split("T")[0]
        version = run.get("version", "v1")

        rows.append(f'''
            <tr onclick="togglePlots({i}, '{version}')">
                <td>{version}{badge}</td>
                <td>{timestamp}</td>
                <td>{run.get("threshold", "N/A")}</td>
                <td>{test_m.get("precision", 0):.4f}</td>
                <td>{test_m.get("recall", 0):.4f}</td>
                <td>{test_m.get("f1", 0):.4f}</td>
                <td>{test_m.get("roc_auc", 0):.4f}</td>
                <td><span class="expand-icon" id="expand-{i}"><svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg></span></td>
            </tr>
            <tr id="plots-{i}" class="plot-row">
                <td colspan="8">
                    <div id="wrapper-{i}" class="history-charts-wrapper">
                        <div id="plot-container-{i}" class="history-charts-container"></div>
                    </div>
                </td>
            </tr>
        ''')
    return "".join(rows)


def _render_empty_or_table(history: list, rows_html: str) -> str:
    if not history:
        return '<p style="color: var(--text-muted);">No training history available yet. Run <code>python run_training.py</code> to generate metrics.</p>'
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
                    <th></th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
