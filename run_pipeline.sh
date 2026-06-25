#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Finish Phase 1: run the at-scale labeling pipeline in one shot.
# Prerequisite: a .env file containing GROQ_API_KEY (see .env.example).
#   cp .env.example .env   &&   open -e .env   # paste your rotated Groq key
# Then:  bash run_pipeline.sh
# ---------------------------------------------------------------------------
set -e
cd "$(dirname "$0")"

if [ ! -f .env ] || ! grep -qE '^GROQ_API_KEY=gsk' .env; then
  echo "✗ No GROQ_API_KEY in .env. Run: cp .env.example .env  then paste your key."
  exit 1
fi

# Optional: if real Reddit creds are present, collect Reddit + rebuild corpus first.
if grep -qE '^REDDIT_CLIENT_ID=[^[:space:]]' .env && ! grep -q 'your_client_id' .env; then
  echo "==> Reddit creds found — collecting Reddit and rebuilding corpus"
  python3 -m pip install --quiet praw 2>/dev/null || true
  python3 collect_reddit.py
  python3 build_dataset.py
fi

echo "==> [1/2] Labeling ~1,002 reviews at scale (verbatim guardrail on) ..."
python3 label_reviews.py

echo "==> [2/2] Computing grounded insights ..."
python3 analyze.py

echo ""
echo "✅ Done. View the dashboard:"
echo "   python3 -m streamlit run app.py"
