#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
serve.py — خادم تجربة المتصفح على الشبكة المحلية (للهاتف) + HTTPS اختياري
============================================================================
مبني بمكتبات بايثون المدمجة فقط (يعمل على Windows بلا أي تثبيت).

    python serve.py                    ← http://<IP>:8777 (كل الواجهات)
    python serve.py --port 9000
    python serve.py --https            ← يولّد شهادة محلية إن لم توجد
    python serve.py --https --cert c.pem --key k.pem
    python serve.py --host 127.0.0.1   ← جهازك فقط (افتراضي: 0.0.0.0)

ملاحظات:
  - يرسل ترويسات COOP/COEP — تفعيل SharedArrayBuffer لخيوط WASM المتعددة
    (بدونها يعمل التطبيق بخيط واحد — أبطأ لكن سليم).
  - HTTPS يلزم لتجربة WebGPU من الهاتف على عنوان IP (سياق آمن) —
    الشهادة الذاتية تتطلب خطوة ثقة إضافية على الهاتف (انظر README).
  - إيقاف الخادم: Ctrl+C في نافذة الطرفية.
"""
import argparse
import os
import socket
import ssl
import subprocess
import sys
import threading

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

MIME = {
    '.wasm': 'application/wasm',
    '.mjs': 'text/javascript',
    '.js': 'text/javascript',
    '.json': 'application/json',
    '.onnx': 'application/octet-stream',
    '.html': 'text/html; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.wav': 'audio/wav',
    '.png': 'image/png',
    '.svg': 'image/svg+xml',
}


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # عزل المصدر — يفعّل SharedArrayBuffer (خيوط WASM المتعددة)
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        # لا تخزين مؤقت أثناء التطوير (نماذج كبيرة تُنزَّل كل مرة عند التعديل)
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def guess_type(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in MIME:
            return MIME[ext]
        return super().guess_type(path)

    def log_message(self, fmt, *args):                 # noqa: N802
        sys.stdout.write('  %s - %s\n' % (self.address_string(), fmt % args))
        sys.stdout.flush()


def lan_ip():
    """عنوان IP الفعلي للحاسوب على الشبكة (مهّرب من ترتيب الواجهات)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))          # لا يُرسل شيئًا فعليًا
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


def ensure_cert(cert, key):
    """توليد شهادة محلية موقّعة ذاتيًا عبر openssl إن توفر (خيار --https)."""
    if os.path.exists(cert) and os.path.exists(key):
        return True
    cert_dir = os.path.dirname(os.path.abspath(cert))
    os.makedirs(cert_dir, exist_ok=True)
    print(f'[https] توليد شهادة محلية: {cert}')
    # SAN يغطي عنوان IP (مطلوب للمتصفحات الحديثة)
    ip = lan_ip()
    cmd = ['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
           '-keyout', key, '-out', cert, '-days', '825',
           '-subj', '/CN=eiqaz-local',
           '-addext', f'subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost']
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        if r.returncode == 0:
            return True
        print('[https] فشل openssl:', r.stderr.strip()[-300:])
    except FileNotFoundError:
        print('[https] openssl غير موجود في PATH.')
    except Exception as e:                              # noqa: BLE001
        print('[https] فشل التوليد:', e)
    return False


def main():
    ap = argparse.ArgumentParser(description='خادم تجربة إيقاز (LAN/HTTPS)')
    ap.add_argument('--host', default='0.0.0.0',
                    help='العنوان (افتراضي 0.0.0.0 — كل الواجهات، للهاتف)')
    ap.add_argument('--port', type=int, default=8777)
    ap.add_argument('--https', action='store_true',
                    help='تشغيل HTTPS (مطلوب لWebGPU من الهاتف عبر IP)')
    ap.add_argument('--cert', default=os.path.join(HERE, 'cert', 'local.pem'))
    ap.add_argument('--key', default=os.path.join(HERE, 'cert', 'local.key'))
    args = ap.parse_args()

    os.chdir(HERE)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)

    scheme = 'http'
    if args.https:
        ok = ensure_cert(args.cert, args.key)
        if not ok:
            print('[https] تعذر تجهيز الشهادة — أكمل عبر HTTP أو ثبّت openssl'
                  ' أو استخدم mkcert (انظر README).')
            sys.exit(1)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(args.cert, args.key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = 'https'

    ip = lan_ip()
    print('=' * 62)
    print('  تجربة محرك إيقاز داخل المتصفح')
    print('=' * 62)
    print(f'  من الحاسوب   : {scheme}://localhost:{args.port}')
    if args.host == '0.0.0.0':
        print(f'  من الهاتف    : {scheme}://{ip}:{args.port}')
        print('                (الهاتف على نفس شبكة Wi-Fi)')
        print(f'  عنوان IP للحاسوب: {ip}  (ipconfig في Windows للتأكد)')
    if args.https:
        print('  HTTPS بشهادة محلية — المتصفح سيعرض تحذيرًا أول مرة')
        print('  (تفاصيل القبول على الهاتف في README.md §تشغيل-HTTPS)')
    print('  إيقاف الخادم: Ctrl+C')
    print('=' * 62)
    if not args.https:
        print('  ملاحظة: لتجربة WebGPU من الهاتف استخدم --https')

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n[serve] أُوقف الخادم.')


if __name__ == '__main__':
    main()
