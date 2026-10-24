
import os
import requests
from dotenv import load_dotenv

load_dotenv()


def build_markdown_report(df, profile, scores=None, task_type=None):
    lines = [
        "# AutoInsight AI — Data Analysis Report",
        "",
        "## Dataset Overview",
        f"- Rows: {profile['rows']}",
        f"- Columns: {profile['columns']}",
        f"- Missing cells: {profile['missing_cells']}",
        f"- Duplicate rows: {profile['duplicate_rows']}",
        f"- Approximate memory usage: {profile['memory_mb']} MB",
        "",
        "## Data Quality",
        profile["quality"].to_markdown(index=False),
        "",
    ]

    if not profile["numeric_summary"].empty:
        lines.extend([
            "## Numeric Summary",
            profile["numeric_summary"].to_markdown(),
            "",
        ])

    if scores is not None and not scores.empty:
        lines.extend([
            f"## Model Comparison ({task_type})",
            scores.to_markdown(index=False),
            "",
        ])

    lines.extend([
        "## Limitations",
        "- Anomaly detection identifies unusual records, not confirmed errors.",
        "- Model scores depend on the selected target and test split.",
        "- Correlation does not prove causation.",
        "- Review data quality and model suitability before decisions.",
    ])

    return "\n".join(lines)


def optional_llm_explanation(summary):
    api_key = os.getenv("LLM_API_KEY", "").strip()
    api_url = os.getenv("LLM_API_URL", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()

    if not (api_key and api_url and model):
        return None, (
            "LLM is not configured. Add LLM_API_KEY, "
            "LLM_API_URL, and LLM_MODEL to your .env file."
        )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful data science assistant. "
                    "Explain only the supplied aggregate statistics. "
                    "Do not invent findings. Mention limitations."
                ),
            },
            {
                "role": "user",
                "content": summary[:12000],
            },
        ],
        "temperature": 0.2,
    }

    try:
        response = requests.post(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=45,
        )
        response.raise_for_status()

        data = response.json()
        explanation = data["choices"][0]["message"]["content"]

        return str(explanation), None

    except Exception as exc:
        return None, f"LLM request failed: {exc}"

from pathlib import Path


def build_markdown_report(df, profile, scores=None, task_type=None):
    quality = profile.get(
        "column_quality",
        profile.get("quality"),
    )

    lines = [
        "# AutoInsight AI — Data Analysis Report",
        "",
        "## Dataset Overview",
        f"- Rows: {profile['rows']}",
        f"- Columns: {profile['columns']}",
        f"- Missing cells: {profile['missing_cells']}",
        f"- Duplicate rows: {profile['duplicate_rows']}",
        f"- Approximate memory usage: {profile['memory_mb']} MB",
        "",
        "## Data Quality",
    ]

    if quality is not None and not quality.empty:
        lines.append(quality.to_markdown(index=False))
    else:
        lines.append("No column quality details available.")

    numeric_summary = profile.get("numeric_summary")
    if numeric_summary is not None and not numeric_summary.empty:
        lines.extend([
            "",
            "## Numeric Summary",
            numeric_summary.to_markdown(),
        ])

    categorical_summary = profile.get("categorical_summary")
    if categorical_summary is not None and not categorical_summary.empty:
        lines.extend([
            "",
            "## Categorical Summary",
            categorical_summary.to_markdown(),
        ])

    if scores is not None:
        lines.extend([
            "",
            "## Model Comparison",
            f"- Task type: {task_type or 'Not specified'}",
            "",
            scores.to_markdown(index=False),
        ])

    lines.extend([
        "",
        "## Limitations",
        "- Findings depend on the supplied dataset and selected features.",
        "- Anomalies are candidates for investigation, not confirmed errors.",
        "- Model scores do not guarantee performance on future data.",
        "",
    ])

    return "\n".join(lines)


def optional_llm_explanation(summary):
    """Generate an optional explanation using a configured API."""
    import os
    import requests
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("LLM_API_KEY", "").strip()
    api_url = os.getenv("LLM_API_URL", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()

    if not api_key or not api_url or not model:
        return None, "LLM API is not configured. Core features work without it."

    try:
        response = requests.post(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Explain data analysis findings clearly. "
                            "Do not invent facts or claim causation."
                        ),
                    },
                    {"role": "user", "content": str(summary)},
                ],
                "temperature": 0.2,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        explanation = data["choices"][0]["message"]["content"]
        return explanation, None
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return None, f"Could not generate AI explanation: {exc}"
