import os
import multiprocessing

# ==========================================
# Timeout & Performance Settings
# ==========================================

timeout = 300
keepalive = 5

# ==========================================
# Concurrency & Worker Settings
# ==========================================

workers = 2
threads = 4
worker_class = 'gthread'

max_requests = 1000
max_requests_jitter = 50

# ==========================================
# Network & Binding Settings
# ==========================================

bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"

# ==========================================
# Logging Settings
# ==========================================

accesslog = '-'
errorlog = '-'
loglevel = 'info'
