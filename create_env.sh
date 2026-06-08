#!/bin/bash
# =============================================================================
# BRZRKR — Interactive .env Generator
# Run this script to create your .env file interactively.
# =============================================================================

set -e

ENV_FILE=".env"
TEMPLATE=".env.template"

echo "====================================================================="
echo "  BRZRKR Multi-Account .env Generator"
echo "====================================================================="
echo ""
echo "This will create $ENV_FILE from the template."
echo "You'll be prompted for each required value."
echo "Press Enter to skip optional values."
echo ""

if [[ -f "$ENV_FILE" ]]; then
    read -p "$ENV_FILE already exists. Overwrite? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Function to read secret (hidden input)
read_secret() {
    local prompt="$1"
    local var_name="$2"
    local value=""
    while [[ -z "$value" ]]; do
        read -s -p "$prompt: " value
        echo
        if [[ -z "$value" ]]; then
            echo "  (required - cannot be empty)"
        fi
    done
    printf "%s=%s\n" "$var_name" "$value"
}

# Function to read optional value
read_optional() {
    local prompt="$1"
    local var_name="$2"
    local default="$3"
    read -p "$prompt [$default]: " value
    if [[ -z "$value" ]]; then
        value="$default"
    fi
    if [[ -n "$value" ]]; then
        printf "%s=%s\n" "$var_name" "$value"
    fi
}

{
    echo "# BRZRKR — Auto-generated $(date)"
    echo "# ============================================================================="
    echo ""
    echo "# --- Alpaca Accounts (REQUIRED for each account you want to use) ---"
    echo ""

    # Primary (required)
    echo "# Primary account (40% allocation)"
    read_secret "ALPACA_API_KEY_PRIMARY" "ALPACA_API_KEY_PRIMARY"
    read_secret "ALPACA_SECRET_KEY_PRIMARY" "ALPACA_SECRET_KEY_PRIMARY"

    # Secondary (required)
    echo ""
    echo "# Secondary account (25% allocation)"
    read_secret "ALPACA_API_KEY_SECONDARY" "ALPACA_API_KEY_SECONDARY"
    read_secret "ALPACA_SECRET_KEY_SECONDARY" "ALPACA_SECRET_KEY_SECONDARY"

    # Tertiary (required)
    echo ""
    echo "# Tertiary account (20% allocation)"
    read_secret "ALPACA_API_KEY_TERTIARY" "ALPACA_API_KEY_TERTIARY"
    read_secret "ALPACA_SECRET_KEY_TERTIARY" "ALPACA_SECRET_KEY_TERTIARY"

    # Optional accounts
    echo ""
    echo "# Optional 4th account (crypto) — press Enter to skip"
    read -p "Add crypto account? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read_secret "ALPACA_API_KEY_CRYPTO" "ALPACA_API_KEY_CRYPTO"
        read_secret "ALPACA_SECRET_KEY_CRYPTO" "ALPACA_SECRET_KEY_CRYPTO"
    fi

    echo ""
    echo "# Optional 5th account (futures proxy) — press Enter to skip"
    read -p "Add futures proxy account? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read_secret "ALPACA_API_KEY_FUTURES" "ALPACA_API_KEY_FUTURES"
        read_secret "ALPACA_SECRET_KEY_FUTURES" "ALPACA_SECRET_KEY_FUTURES"
    fi

    # Live money toggle
    echo ""
    echo "# --- Live Money Guard ---"
    echo "# IMPORTANT: Real money requires BOTH this flag AND passing the promotion gate"
    read_optional "Enable ALPACA_LIVE=true? (true/false)" "ALPACA_LIVE" "false"

    # Data APIs (optional)
    echo ""
    echo "# --- Data API Keys (optional but improve signal quality) ---"
    read_optional "Polygon.io API key" "POLYGON_API_KEY" ""
    read_optional "Tradier API key" "TRADIER_API_KEY" ""
    read_optional "Unusual Whales API key" "UNUSUAL_WHALES_API_KEY" ""
    read_optional "NewsAPI.org key" "NEWSAPI_KEY" ""
    read_optional "FRED API key" "FRED_API_KEY" ""
    read_optional "Finnhub API key" "FINNHUB_API_KEY" ""
    read_optional "Alpha Vantage API key" "ALPHA_VANTAGE_API_KEY" ""
    read_optional "Hugging Face token" "HF_TOKEN" ""

    # Telegram
    echo ""
    echo "# --- Telegram Alerts ---"
    read_optional "Telegram Bot Token" "TELEGRAM_BOT_TOKEN" "8691113486:AAFkG3m707BLBOlRE29OjipeTdbKMYnAQZo"

    # Python path
    echo ""
    echo "# --- Environment ---"
    echo "PYTHONPATH=/Users/max51/BRZRKR trader"
} > "$ENV_FILE"

chmod 600 "$ENV_FILE"
echo ""
echo "====================================================================="
echo "  Done! Created $ENV_FILE with 600 permissions (owner read/write only)"
echo "====================================================================="
echo ""
echo "To use:"
echo "  source .env"
echo "  python agent.py --dry-run --broker multi_account"
echo ""
echo "To verify keys are loaded:"
echo "  python -c \"import os; print('Primary:', 'OK' if os.getenv('ALPACA_API_KEY_PRIMARY') else 'MISSING')\""