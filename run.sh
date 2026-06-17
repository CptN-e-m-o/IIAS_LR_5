#!/usr/bin/env bash
set -euo pipefail
python src/run_experiments.py --data data/customer_support_tickets.csv --out outputs
