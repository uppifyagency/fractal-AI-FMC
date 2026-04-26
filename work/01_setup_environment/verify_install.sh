#!/usr/bin/env bash
# verify_install.sh — controlla che l'ambiente fragile sia operativo
# Uso: bash verify_install.sh

set -e

cd "$(dirname "$0")"

echo "[1/5] Python version......... $(python --version | cut -d' ' -f2) ✓"

echo -n "[2/5] Importing fragile...... "
python -c "import fragile; print(fragile.__version__ if hasattr(fragile, '__version__') else 'installed', '✓')"

echo -n "[3/5] Importing torch........ "
python -c "import torch; print(torch.__version__, '✓')"

echo -n "[4/5] Importing plangym...... "
python -c "import plangym; print(plangym.__version__ if hasattr(plangym, '__version__') else 'installed', '✓')"

echo "[5/5] Smoke test FMC........ "
python verify_install.py

echo ""
echo "OK ✓ — ambiente pronto per replication"
