#!/bin/bash
# view_splat.sh — open a .splat file in the local browser viewer
#
# Usage:
#   bash view_splat.sh <path/to/file.splat>

set -e

SPLAT="$1"
REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/home/communications/miniconda3/envs/instantsplat/bin/python
PORT=9999

if [[ -z "$SPLAT" ]]; then
    echo "Usage: bash view_splat.sh <path/to/file.splat>"
    exit 1
fi

if [[ ! -f "$SPLAT" ]]; then
    echo "Error: file not found: $SPLAT"
    exit 1
fi

SPLAT_ABS="$(realpath "$SPLAT")"
SPLAT_DIR="$(dirname "$SPLAT_ABS")"
SPLAT_NAME="$(basename "$SPLAT_ABS")"

# Kill anything already on that port
fuser -k ${PORT}/tcp 2>/dev/null || true

# CORS-enabled server that also serves viewer.html from the repo root
"$PYTHON" - "$SPLAT_DIR" "$REPO" "$PORT" <<'EOF' &
import sys, http.server, socketserver, os

splat_dir = sys.argv[1]
repo_dir  = sys.argv[2]
port      = int(sys.argv[3])

class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        # viewer.html served from repo root; everything else from splat dir
        if path.rstrip('/') == '' or path == '/viewer.html':
            return os.path.join(repo_dir, 'viewer.html')
        return os.path.join(splat_dir, path.lstrip('/'))

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, *args):
        pass

with socketserver.TCPServer(("", port), Handler) as httpd:
    httpd.serve_forever()
EOF

SERVER_PID=$!
sleep 0.5

URL="http://localhost:${PORT}/viewer.html?splat=${SPLAT_NAME}"
echo "Serving: $SPLAT_ABS"
echo "Opening: $URL"
echo ""
echo "Press Ctrl+C to stop the server."

xdg-open "$URL" 2>/dev/null || echo "Open manually: $URL"

wait "$SERVER_PID"
