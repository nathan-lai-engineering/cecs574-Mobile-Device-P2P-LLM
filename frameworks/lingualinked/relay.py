"""
TCP relay: bidirectional bridge between Android emulator and A1 worker node.
Run on Windows laptop before starting inference.

Usage: python relay.py --a1 <A1_TAILSCALE_IP>

Port routing:
  12346 -> A1:12346     Android->A1: header tensor (Android connects to 10.0.2.2:12346)
  55555 -> A1:55555     Android->A1: bandwidth test
  12348 -> 127.0.0.1:12347  A1->Android: tailer pulls from header
                            Requires: adb forward tcp:12347 tcp:12346
"""
import socket
import threading
import time
import argparse

PORTS = [12346, 55555]       # Android->A1: forward to a1_ip on same port
ANDROID_RELAY_PORT = 12348   # A1->Android: A1 connects here
ANDROID_ADB_PORT   = 12347   # adb forward tcp:12347 tcp:12346 must be running


def pipe(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except Exception:
        pass
    finally:
        try: src.close()
        except Exception: pass
        try: dst.close()
        except Exception: pass


def set_keepalive(sock):
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    try:  # Linux-specific tuning (ignored on Windows)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE,  10)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT,    5)
    except AttributeError:
        pass


def handle(client, a1_ip, port):
    set_keepalive(client)
    server = None
    for attempt in range(30):  # retry for up to 60s (30 x 2s)
        try:
            server = socket.create_connection((a1_ip, port), timeout=2)
            break
        except Exception:
            time.sleep(2)
    if server is None:
        print(f"[relay] Could not connect to {a1_ip}:{port} after retries")
        client.close()
        return
    set_keepalive(server)
    print(f"[relay] {client.getpeername()} <-> {a1_ip}:{port}")
    t1 = threading.Thread(target=pipe, args=(client, server), daemon=True)
    t2 = threading.Thread(target=pipe, args=(server, client), daemon=True)
    t1.start()
    t2.start()


def listen(port, dest_ip, dest_port=None):
    if dest_port is None:
        dest_port = port
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(20)
    print(f"[relay] Listening on 0.0.0.0:{port} -> {dest_ip}:{dest_port}")
    while True:
        client, _ = srv.accept()
        threading.Thread(target=handle, args=(client, dest_ip, dest_port), daemon=True).start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", required=True, help="A1 Tailscale IP, e.g. 100.110.117.116")
    args = parser.parse_args()

    for port in PORTS:
        threading.Thread(target=listen, args=(port, args.a1), daemon=True).start()

    # Reverse relay: A1 connects here to reach Android's ZMQ ROUTER
    # Requires: adb forward tcp:12347 tcp:12346  (run on laptop before starting)
    threading.Thread(
        target=listen,
        args=(ANDROID_RELAY_PORT, "127.0.0.1", ANDROID_ADB_PORT),
        daemon=True,
    ).start()

    print("[relay] Running. Press Ctrl+C to stop.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("[relay] Stopped.")


if __name__ == "__main__":
    main()
