#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  EMAX TRADING PLATFORM — FIRST TIME SETUP
#  Run this once on a fresh Ubuntu 24 droplet
#  Usage: bash setup.sh
# ═══════════════════════════════════════════════════════════

set -e  # stop on any error

REPO_URL="https://github.com/Hota5/T.V.W.git"
APP_DIR="/opt/trader"
SERVICE_NAME="trader"

# ── Colors ──
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${CYAN}[SETUP]${NC} $1"; }
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     EMAX TRADING PLATFORM — SETUP       ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

# ── Check running as root ──
[ "$EUID" -ne 0 ] && fail "Run as root: sudo bash setup.sh"

# ── Step 1: System packages ──
log "Installing system packages..."
apt update -qq
apt install -y git python3-pip python3-venv nginx ufw curl > /dev/null
ok "System packages installed"

# ── Step 2: Clone repo ──
log "Cloning repository..."
if [ -d "$APP_DIR" ]; then
    warn "$APP_DIR already exists — pulling latest instead"
    cd "$APP_DIR"
    git pull
else
    git clone "$REPO_URL" "$APP_DIR"
fi
ok "Repository ready at $APP_DIR"

# ── Step 3: Python venv + dependencies ──
log "Setting up Python environment..."
cd "$APP_DIR"
python3 -m venv venv
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
deactivate
ok "Python environment ready"

# ── Step 4: Generate secret key ──
log "Generating secret key..."
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
ok "Secret key generated"

# ── Step 5: Create systemd service ──
log "Creating systemd service..."
cat > /etc/systemd/system/${SERVICE_NAME}.service << EOF
[Unit]
Description=EMAX Trading Platform
After=network.target

[Service]
User=root
WorkingDirectory=${APP_DIR}
Environment="SECRET_KEY=${SECRET_KEY}"
ExecStart=${APP_DIR}/venv/bin/gunicorn -w 2 -b 127.0.0.1:5000 --timeout 600 app:app
Restart=always
RestartSec=10
StandardOutput=append:${APP_DIR}/service.log
StandardError=append:${APP_DIR}/service.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ${SERVICE_NAME}
systemctl start ${SERVICE_NAME}
sleep 2

if systemctl is-active --quiet ${SERVICE_NAME}; then
    ok "Service started and enabled"
else
    fail "Service failed to start — check: journalctl -u ${SERVICE_NAME} -n 30"
fi

# ── Step 6: Nginx config ──
log "Configuring Nginx..."
cat > /etc/nginx/sites-available/${SERVICE_NAME} << 'EOF'
server {
    listen 80;
    server_name _;

    proxy_read_timeout    1800;
    proxy_connect_timeout 60;
    proxy_send_timeout    1800;
    proxy_send_timeout    1800;

    client_max_body_size 4G;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_buffering off;
    }
}
EOF

ln -sf /etc/nginx/sites-available/${SERVICE_NAME} /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t > /dev/null 2>&1 && systemctl restart nginx
ok "Nginx configured"

# ── Step 7: Firewall ──
log "Configuring firewall..."
ufw allow 223  > /dev/null 2>&1 || true
ufw allow 80   > /dev/null 2>&1 || true
ufw allow 22   > /dev/null 2>&1 || true
ufw --force enable > /dev/null
ok "Firewall configured (ports 80, 22, 223 open)"

# ── Step 8: Create update script ──
log "Installing update script..."
cp "${APP_DIR}/update.sh" /usr/local/bin/update-trader 2>/dev/null || true
chmod +x /usr/local/bin/update-trader 2>/dev/null || true

# ── Done ──
IP=$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           SETUP COMPLETE ✓               ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Platform URL:  ${CYAN}http://${IP}${NC}"
echo -e "  Default login: ${YELLOW}admin / changeme123${NC}"
echo -e "  App directory: ${APP_DIR}"
echo -e "  Logs:          ${APP_DIR}/service.log"
echo ""
echo -e "  ${RED}!! Change your password immediately after first login !!${NC}"
echo ""
echo -e "  To update in future:  ${CYAN}update-trader${NC}"
echo -e "  To view logs:         ${CYAN}journalctl -u trader -f${NC}"
echo -e "  To restart:           ${CYAN}systemctl restart trader${NC}"
echo ""
