# ═══════════════════════════════════════════════════════════
#  EMA CROSSOVER PLATFORM — DIGITAL OCEAN SETUP GUIDE
# ═══════════════════════════════════════════════════════════
#  This is a step-by-step guide. Every command is copy-paste.
#  Takes about 10–15 minutes to set up from scratch.
# ═══════════════════════════════════════════════════════════


# ─────────────────────────────────────────────
# STEP 1 — Create a Droplet on DigitalOcean
# ─────────────────────────────────────────────

1. Log in to DigitalOcean → Create → Droplets
2. Choose:
   - Image:  Ubuntu 24.04 LTS
   - Size:   Basic / Regular → $6/mo (1 vCPU, 1 GB RAM) — enough for the bot
             For faster optimization → $12/mo (2 vCPU, 2 GB RAM)
   - Region: Pick closest to you (e.g. Amsterdam, Frankfurt)
   - Auth:   SSH key (recommended) or Password
3. Click Create Droplet
4. Copy the IP address shown (e.g. 123.45.67.89)


# ─────────────────────────────────────────────
# STEP 2 — Connect to your droplet
# ─────────────────────────────────────────────

# On your computer (Windows: use PuTTY or Windows Terminal):
ssh root@YOUR_IP_ADDRESS

# If asked "Are you sure you want to continue?" → type: yes


# ─────────────────────────────────────────────
# STEP 3 — Install everything (copy-paste all at once)
# ─────────────────────────────────────────────

apt update && apt upgrade -y
apt install -y python3-pip python3-venv nginx ufw


# ─────────────────────────────────────────────
# STEP 4 — Upload your files
# ─────────────────────────────────────────────

# On your LOCAL computer (not the server), run:
scp -r /path/to/trader/ root@YOUR_IP_ADDRESS:/opt/trader/

# Or if you're on Windows, use WinSCP (free) to drag and drop the trader/ folder to /opt/trader/
# The trader/ folder should contain: app.py, index.html, requirements.txt


# ─────────────────────────────────────────────
# STEP 5 — Install Python dependencies (back on server)
# ─────────────────────────────────────────────

cd /opt/trader
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate


# ─────────────────────────────────────────────
# STEP 6 — Create a systemd service (runs forever, auto-restarts)
# ─────────────────────────────────────────────

# Paste this entire block (it creates the service file):
cat > /etc/systemd/system/trader.service << 'EOF'
[Unit]
Description=EMA Crossover Trading Platform
After=network.target

[Service]
User=root
WorkingDirectory=/opt/trader
# First generate a secret key (run this command and copy the output):
python3 -c "import secrets; print(secrets.token_hex(32))"

# Paste the output in SECRET_KEY= below:
Environment="SECRET_KEY=PASTE_YOUR_SECRET_KEY_HERE"
ExecStart=/opt/trader/venv/bin/gunicorn -w 2 -b 127.0.0.1:5000 --timeout 300 app:app
Restart=always
RestartSec=10
StandardOutput=append:/opt/trader/service.log
StandardError=append:/opt/trader/service.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable trader
systemctl start trader

# Check it started OK:
systemctl status trader


# ─────────────────────────────────────────────
# STEP 7 — Set up Nginx (so you can use http://YOUR_IP)
# ─────────────────────────────────────────────

cat > /etc/nginx/sites-available/trader << 'EOF'
server {
    listen 80;
    server_name _;

    # Increase timeouts for long backtest/optimization runs
    proxy_read_timeout 600;
    proxy_connect_timeout 600;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
    }
}
EOF

ln -sf /etc/nginx/sites-available/trader /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx


# ─────────────────────────────────────────────
# STEP 8 — Open firewall
# ─────────────────────────────────────────────

ufw allow 22    # SSH
ufw allow 80    # HTTP
ufw --force enable


# ─────────────────────────────────────────────
# STEP 9 — Open in browser
# ─────────────────────────────────────────────

# Go to:  http://YOUR_IP_ADDRESS
# You will see a login page.
# Default credentials:  username: admin   password: changeme123
# !! Go to Settings tab immediately and change your password !!


# ═══════════════════════════════════════════════════════════
#  USEFUL COMMANDS (for later)
# ═══════════════════════════════════════════════════════════

# Restart after updating files:
systemctl restart trader

# View live logs:
journalctl -u trader -f

# View app logs:
tail -f /opt/trader/service.log

# Update files from your computer:
scp app.py index.html root@YOUR_IP:/opt/trader/
systemctl restart trader

# Check if it's running:
systemctl status trader


# ═══════════════════════════════════════════════════════════
#  OPTIONAL — Add a domain name (e.g. trader.yourdomain.com)
# ═══════════════════════════════════════════════════════════

# 1. In your domain registrar, add an A record:
#    trader.yourdomain.com → YOUR_IP

# 2. Install certbot for free HTTPS:
apt install -y certbot python3-certbot-nginx
certbot --nginx -d trader.yourdomain.com

# That gives you https://trader.yourdomain.com


# ═══════════════════════════════════════════════════════════
#  SECURITY NOTE
# ═══════════════════════════════════════════════════════════
# Right now the platform has no login. Anyone with your IP
# can access it. To add basic password protection:

apt install -y apache2-utils
htpasswd -c /etc/nginx/.htpasswd yourname
# (enter your password when prompted)

# Then add these two lines inside the location {} block in nginx config:
#   auth_basic "Trading Platform";
#   auth_basic_user_file /etc/nginx/.htpasswd;

nano /etc/nginx/sites-available/trader
# (add the two auth lines, save with Ctrl+X → Y → Enter)
systemctl restart nginx
