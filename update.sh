#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  EMAX TRADING PLATFORM — UPDATE
#  Pulls latest code from GitHub and restarts the service
#  Usage: bash update.sh   OR   update-trader (after setup)
# ═══════════════════════════════════════════════════════════

APP_DIR="/opt/trader"
SERVICE_NAME="trader"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${CYAN}[UPDATE]${NC} $1"; }
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     EMAX TRADING PLATFORM — UPDATE      ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

[ "$EUID" -ne 0 ] && fail "Run as root: sudo bash update.sh"
[ ! -d "$APP_DIR" ] && fail "$APP_DIR not found — run setup.sh first"

cd "$APP_DIR"

# ── Show current version ──
BEFORE=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
log "Current commit: $BEFORE"

# ── Pull latest ──
log "Pulling latest code from GitHub..."
git fetch origin
git reset --hard origin/main 2>/dev/null || git reset --hard origin/master
AFTER=$(git rev-parse --short HEAD)

if [ "$BEFORE" = "$AFTER" ]; then
    warn "Already up to date (commit: $AFTER) — restarting anyway"
else
    ok "Updated $BEFORE → $AFTER"
    # Show what changed
    echo ""
    echo -e "${CYAN}Changes:${NC}"
    git log --oneline ${BEFORE}..${AFTER} 2>/dev/null | head -10 | sed 's/^/  /'
    echo ""
fi

# ── Update Python dependencies if requirements.txt changed ──
if git diff --name-only ${BEFORE} ${AFTER} 2>/dev/null | grep -q "requirements.txt"; then
    log "requirements.txt changed — updating dependencies..."
    source venv/bin/activate
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    deactivate
    ok "Dependencies updated"
else
    log "No dependency changes"
fi


# ── Patch nginx for large uploads if needed ──
if ! grep -q "4G" /etc/nginx/sites-available/trader 2>/dev/null; then
    log "Updating nginx for large file uploads..."
    sed -i 's/client_max_body_size.*/client_max_body_size 4G;/' /etc/nginx/sites-available/trader
    sed -i 's/proxy_read_timeout.*/proxy_read_timeout    1800;/' /etc/nginx/sites-available/trader
    sed -i 's/proxy_send_timeout.*/proxy_send_timeout    1800;/' /etc/nginx/sites-available/trader
    nginx -t > /dev/null 2>&1 && systemctl reload nginx
    ok "Nginx updated"
fi

# ── Restart service ──
log "Restarting service..."
systemctl restart ${SERVICE_NAME}
sleep 2

if systemctl is-active --quiet ${SERVICE_NAME}; then
    ok "Service restarted successfully"
else
    fail "Service failed to restart — check: journalctl -u ${SERVICE_NAME} -n 30"
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           UPDATE COMPLETE ✓              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Commit:  ${CYAN}$AFTER${NC}"
echo -e "  Status:  ${GREEN}Running${NC}"
echo -e "  Logs:    ${CYAN}journalctl -u trader -f${NC}"
echo ""
