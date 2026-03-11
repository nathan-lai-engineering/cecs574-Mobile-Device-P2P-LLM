#!/usr/bin/env python3
"""
LinguaLinked Python Device Simulator

Simulates an Android device participating in LinguaLinked distributed inference.
Supports both header and worker roles, allowing role swapping between sessions.

Usage:
  # Worker node:
  python device_client.py --role worker --ip 127.0.0.2 --server 172.17.0.2

  # Header node (drives inference, has chat prompt):
  python device_client.py --role header --ip 127.0.0.3 --server 172.17.0.2 --model llama-2-7b

  # If another device is behind ADB forward (e.g. Android emulator at 10.0.2.15
  # accessible via adb forward tcp:12346 tcp:12346):
  python device_client.py --role worker --ip 127.0.0.2 --server 172.17.0.2 \\
      --ip-map 10.0.2.15=127.0.0.1

Protocol phases (matching root_server.py):
  RegisterIP → Ready → Open → Prepare → Initialized → Start → Running → Finish → Close
"""

import argparse
import json
import os
import threading
import time
import zipfile
from pathlib import Path

import zmq

try:
    import numpy as np
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("WARNING: onnxruntime/numpy not installed. Running in passthrough mode.")

BASE_PORT = 12345   # Config.port in Android app
ROOT_PORT = 23456   # Root server ZMQ port


# ---------------------------------------------------------------------------
# Device Simulator
# ---------------------------------------------------------------------------

class DeviceSimulator:

    def __init__(self, role, local_ip, server_ip, model_name=None,
                 server_port=ROOT_PORT, ip_map=None, model_dir=None):
        self.role = role                    # 'header' or 'worker'
        self.local_ip = local_ip
        self.server_ip = server_ip
        self.model_name = model_name
        self.server_port = server_port
        self.ip_map = ip_map or {}          # e.g. {"10.0.2.15": "127.0.0.1"}

        self.context = zmq.Context()
        self.root_socket = None

        # Populated during Open phase
        self.ip_graph = []
        self.task_type = "generation"
        self.core_pool_size = 2
        self.num_sample = 1
        self.max_length = 40
        self.dependency = {}

        self.device_id = -1
        self.is_header = False
        self.is_tailer = False

        # ONNX sessions (one per model shard assigned to this device)
        self.sessions = []
        self.tokenizer = None

        # Inter-device tensor buffers: sample_id -> bytes
        self.output_data = {}
        self._data_ready = {}       # sample_id -> threading.Event
        self._seq_len = {}          # sample_id -> int (real token count from Android)

        self.model_dir = Path(model_dir) if model_dir else Path(f"./device_models/{local_ip.replace('.', '_')}")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._running = False
        self._prompt_queue = []
        self._prompt_lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------------

    def run(self):
        print(f"\n[{self.local_ip}] ========== LinguaLinked Device Simulator ==========")
        print(f"[{self.local_ip}] Role: {self.role} | Server: {self.server_ip}:{self.server_port}")
        if self.ip_map:
            print(f"[{self.local_ip}] IP map (ADB): {self.ip_map}")

        self.root_socket = self.context.socket(zmq.DEALER)
        self.root_socket.connect(f"tcp://{self.server_ip}:{self.server_port}")

        self._register()
        self._lifecycle()

    # -----------------------------------------------------------------------
    # Phase 1: RegisterIP
    # -----------------------------------------------------------------------

    def _register(self):
        print(f"[{self.local_ip}] Registering as '{self.role}'...")
        payload = {"ip": self.local_ip, "role": self.role}
        if self.role == "header" and self.model_name:
            payload["model"] = self.model_name

        self.root_socket.send_multipart([
            b"RegisterIP",
            json.dumps(payload).encode()
        ])

        # Server sends back: b"True" (need monitor) or b"False" (model cached)
        monitor_msg = self.root_socket.recv_multipart()
        need_monitor = monitor_msg[0].decode().strip() == "True"
        print(f"[{self.local_ip}] Registered. Monitor: {need_monitor}")

        if need_monitor:
            self._run_monitor()

    # -----------------------------------------------------------------------
    # Monitor phase (runs before lifecycle when server sends need_monitor=True)
    # -----------------------------------------------------------------------

    def _run_monitor(self):
        """Connect to monitor port 34567, send hardware metrics, mirror MonitorService.kt."""
        import struct
        import socket as tcp_socket
        MONITOR_PORT = 34567
        BANDWIDTH_PORT = 55555

        # Start TCP bandwidth-test server on port 55555 so Android can connect for measurement.
        # Accept connections and drain any data sent, matching MonitorService.kt's bandwidth test.
        def _bandwidth_server():
            try:
                srv = tcp_socket.socket(tcp_socket.AF_INET, tcp_socket.SOCK_STREAM)
                srv.setsockopt(tcp_socket.SOL_SOCKET, tcp_socket.SO_REUSEADDR, 1)
                srv.bind(("0.0.0.0", BANDWIDTH_PORT))
                srv.listen(5)
                srv.settimeout(120)
                print(f"[{self.local_ip}] BandwidthServer: listening on port {BANDWIDTH_PORT}")
                while True:
                    try:
                        conn, addr = srv.accept()
                        print(f"[{self.local_ip}] BandwidthServer: connection from {addr}")
                        conn.settimeout(10)
                        try:
                            while conn.recv(65536):
                                pass
                        except Exception:
                            pass
                        conn.close()
                    except tcp_socket.timeout:
                        break
                    except Exception:
                        break
                srv.close()
            except Exception as e:
                print(f"[{self.local_ip}] BandwidthServer error: {e}")

        bw_thread = threading.Thread(target=_bandwidth_server, daemon=True)
        bw_thread.start()

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.DEALER)
        sock.connect(f"tcp://{self.server_ip}:{MONITOR_PORT}")
        print(f"[{self.local_ip}] Monitor: connected to {self.server_ip}:{MONITOR_PORT}")

        # Send MonitorIP registration
        payload = json.dumps({"ip": self.local_ip, "role": self.role}).encode()
        sock.send_multipart([b"MonitorIP", payload])
        print(f"[{self.local_ip}] Monitor: sent MonitorIP")

        # Receive IP graph
        ip_graph_bytes = sock.recv()
        ip_graph = ip_graph_bytes.decode()
        print(f"[{self.local_ip}] Monitor: ip_graph = {ip_graph}")

        # Receive flop count (4-byte little-endian uint32)
        flop_bytes = sock.recv()
        num_flop = struct.unpack("<L", flop_bytes)[0] if len(flop_bytes) == 4 else 0
        print(f"[{self.local_ip}] Monitor: num_flop = {num_flop}")

        # Receive flop byte array file (drain chunked transfer)
        self._drain_file_transfer(sock, "flop_byte_array.bin")

        # Receive flop module onnx file
        self._drain_file_transfer(sock, "flop_module.onnx")

        devices = [d for d in ip_graph.split(",") if d]
        n = len(devices)
        my_idx = devices.index(self.local_ip) if self.local_ip in devices else 0

        # Collect and send metrics in response to "continue"/"stop" signals
        # Use poll with timeout so we don't hang if monitor server stalls
        while True:
            if not sock.poll(30000):  # 30s timeout
                print(f"[{self.local_ip}] Monitor: no signal in 30s — exiting monitor")
                break
            signal_raw = sock.recv()
            signal = signal_raw.decode().strip()
            print(f"[{self.local_ip}] Monitor signal: {signal}")
            if signal == "stop":
                break

            # Build dummy but plausible metrics
            latency = [0.0] * n
            for i in range(n):
                if i != my_idx:
                    latency[i] = 5.0 + (abs(i - my_idx) * 2.0)   # ms

            bandwidth = [0.0] * n
            for i in range(n):
                if i != my_idx:
                    bandwidth[i] = 50.0   # MB/s

            import psutil
            vm = psutil.virtual_memory()
            total_mem = vm.total / (1024 * 1024)    # MB
            avail_mem = vm.available / (1024 * 1024)

            flop_speed = 1000.0   # MFLOPS placeholder

            metrics = {
                "ip": self.local_ip,
                "latency": json.dumps(latency),
                "bandwidth": json.dumps(bandwidth),
                "memory": json.dumps([total_mem, avail_mem]),
                "flop": flop_speed,
            }
            sock.send_multipart([b"Monitor", json.dumps(metrics).encode()])
            print(f"[{self.local_ip}] Monitor: sent metrics")

        sock.close()
        print(f"[{self.local_ip}] Monitor: done")

    def _drain_file_transfer(self, sock, label):
        """Receive a chunked file transfer (empty-frame terminated)."""
        total = 0
        while True:
            chunk = sock.recv()
            if chunk == b"":
                break
            total += len(chunk)
        print(f"[{self.local_ip}] Monitor: received {label} ({total:,} bytes)")

    # -----------------------------------------------------------------------
    # Phase 2: Ready → Open → Prepare → Initialized → Start → Running → Finish
    # -----------------------------------------------------------------------

    def _lifecycle(self):
        # Ready
        print(f"[{self.local_ip}] Sending Ready...")
        self.root_socket.send_multipart([b"Ready", self.local_ip.encode()])

        # Open — server sends configuration as 8 separate messages
        # (root_server.py uses individual send_multipart calls, not one big frame)
        first = self.root_socket.recv_multipart()
        if first[0] != b"Open":
            print(f"[{self.local_ip}] ERROR: Expected Open, got {first[0]}")
            return
        open_parts = [first[0]]
        for _ in range(7):
            part = self.root_socket.recv_multipart()
            open_parts.append(part[0])
        self._parse_open_config(open_parts)

        # Prepare — model shard download
        msg = self.root_socket.recv_multipart()
        if msg[0] != b"Prepare":
            print(f"[{self.local_ip}] ERROR: Expected Prepare, got {msg[0]}")
            return
        self._handle_prepare()

        # Load ONNX model
        self._load_model()

        # Initialized
        print(f"[{self.local_ip}] Sending Initialized...")
        self.root_socket.send_multipart([b"Initialized"])

        # Start
        msg = self.root_socket.recv_multipart()
        if msg[0] != b"Start":
            print(f"[{self.local_ip}] ERROR: Expected Start, got {msg[0]}")
            return
        print(f"[{self.local_ip}] Received Start — beginning inference.")

        # Running
        self._running = True
        self.root_socket.send_multipart([b"Running"])
        self._run_inference()

        # Finish
        self._running = False
        print(f"[{self.local_ip}] Sending Finish...")
        self.root_socket.send_multipart([b"Finish"])
        msg = self.root_socket.recv_multipart()
        print(f"[{self.local_ip}] Received {msg[0].decode()}. Done.")

    # -----------------------------------------------------------------------
    # Config parsing (Open message)
    # -----------------------------------------------------------------------

    def _parse_open_config(self, msg):
        # msg = [b'Open', graph, session_index, task_type,
        #        core_pool_size, num_sample, max_length, dependency_json]
        self.ip_graph        = msg[1].decode().split(",")
        session_raw          = msg[2].decode()
        self.task_type       = msg[3].decode()
        self.core_pool_size  = int(msg[4].decode())
        self.num_sample      = int(msg[5].decode())
        self.max_length      = int(msg[6].decode())
        self.dependency      = json.loads(msg[7].decode())

        if self.local_ip in self.ip_graph:
            self.device_id = self.ip_graph.index(self.local_ip)
        else:
            # Coordinator cache may have an old IP — infer position by role
            self.device_id = 0 if self.role == "header" else len(self.ip_graph) - 1
            print(f"[{self.local_ip}] WARNING: IP not in graph {self.ip_graph}, "
                  f"inferring device_id={self.device_id} from role '{self.role}'")
        n               = len(self.ip_graph)
        self.is_header  = (self.device_id == 0)
        self.is_tailer  = (self.device_id == n - 1)

        print(f"[{self.local_ip}] Open config:")
        print(f"  Graph:     {self.ip_graph}")
        print(f"  Device ID: {self.device_id}  (header={self.is_header}, tailer={self.is_tailer})")
        print(f"  Task:      {self.task_type} | samples={self.num_sample} | max_len={self.max_length}")
        print(f"  Sessions:  {session_raw}")

    # -----------------------------------------------------------------------
    # Model download (Prepare phase)
    # -----------------------------------------------------------------------

    def _handle_prepare(self):
        # Receive skip flag
        msg = self.root_socket.recv_multipart()
        skip = msg[0].decode().strip().lower() == "true"

        if skip:
            print(f"[{self.local_ip}] Model already on device — skipping download.")
            return

        print(f"[{self.local_ip}] Downloading model shard...")
        zip_path = self.model_dir / "module.zip"
        total = 0

        with open(zip_path, "wb") as f:
            while True:
                chunk_msg = self.root_socket.recv_multipart()
                chunk = chunk_msg[0]
                if chunk == b"":
                    break
                f.write(chunk)
                total += len(chunk)
                print(f"[{self.local_ip}]   {total:,} bytes received...", end="\r")

        print(f"\n[{self.local_ip}] Download complete: {total:,} bytes")

        print(f"[{self.local_ip}] Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.model_dir)
        print(f"[{self.local_ip}] Extraction complete.")

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------

    def _load_model(self):
        if not ONNX_AVAILABLE:
            print(f"[{self.local_ip}] onnxruntime unavailable — passthrough mode.")
            return

        # Prefer INT8-quantized models if present (e.g. module_int8.onnx over module.onnx)
        all_onnx = sorted(self.model_dir.rglob("*.onnx"))
        # Build a deduplicated list: for each base stem prefer *_int8 variant
        seen_stems = {}
        for p in all_onnx:
            base = p.stem.replace("_int8", "")
            if p.stem.endswith("_int8"):
                seen_stems[base] = p          # int8 wins
            elif base not in seen_stems:
                seen_stems[base] = p
        onnx_files = sorted(seen_stems.values())

        if not onnx_files:
            print(f"[{self.local_ip}] WARNING: No .onnx files in {self.model_dir}. Passthrough mode.")
            return

        ncpu = os.cpu_count() or 4
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = ncpu
        opts.inter_op_num_threads = ncpu
        opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True

        # Prefer AVX2/AVX-512 path on x86-64; fall back to plain CPU
        providers = ["CPUExecutionProvider"]
        try:
            available = ort.get_available_providers()
            for ep in ["OpenVINOExecutionProvider", "DnnlExecutionProvider"]:
                if ep in available:
                    providers.insert(0, ep)
                    break
        except Exception:
            pass

        for path in onnx_files:
            print(f"[{self.local_ip}] Loading {path.name} (providers={providers})...")
            sess = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
            self.sessions.append(sess)
            ins  = [i.name for i in sess.get_inputs()]
            outs = [o.name for o in sess.get_outputs()]
            print(f"[{self.local_ip}]   inputs={ins}  outputs={outs}")

        if self.is_header:
            self._load_tokenizer()

    def _load_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            model_map = {
                "llama-2-7b":  "meta-llama/Llama-2-7b-hf",
                "llama2-7b":   "meta-llama/Llama-2-7b-hf",
                "bloom560m":   "bigscience/bloom-560m",
                "bloom1b1":    "bigscience/bloom-1b1",
                "bloom1b7":    "bigscience/bloom-1b7",
                "bloom3b":     "bigscience/bloom-3b",
            }
            tok_source = self.model_dir if (self.model_dir / "tokenizer.json").exists() \
                         else model_map.get(self.model_name, self.model_name)
            print(f"[{self.local_ip}] Loading tokenizer from {tok_source}...")
            self.tokenizer = AutoTokenizer.from_pretrained(str(tok_source))
            print(f"[{self.local_ip}] Tokenizer ready.")
        except Exception as e:
            print(f"[{self.local_ip}] WARNING: Tokenizer unavailable ({e}). Using char encoding.")

    # -----------------------------------------------------------------------
    # Inference orchestration
    # -----------------------------------------------------------------------

    def _run_inference(self):
        n = len(self.ip_graph)
        j = n   # graph length; used in port formula

        # Launch one P2P thread per core pool slot
        for i in range(self.core_pool_size):
            t = threading.Thread(
                target=self._setup_p2p_sockets, args=(i, j), daemon=True)
            t.start()

        time.sleep(0.5)  # Let sockets bind before connecting

        if self.is_header:
            self._header_inference_loop(j)
        else:
            self._worker_inference_loop(j)

    def _setup_p2p_sockets(self, thread_idx, graph_len):
        """Bind ROUTER sockets for downstream devices to pull from us."""
        n = len(self.ip_graph)

        # Each device binds a ROUTER for every device that will pull from it.
        # Port formula (from Communication.java):
        #   ROUTER port = BASE_PORT + graph_len * thread_idx + (next_id - my_id)

        for next_id in range(self.device_id + 1, n):
            port = BASE_PORT + graph_len * thread_idx + (next_id - self.device_id)
            self._bind_router(port, thread_idx, next_id, graph_len)

        # Tailer also binds a ROUTER for header's return-path DEALER (port offset +1)
        if self.is_tailer:
            port = BASE_PORT + graph_len * thread_idx + 1
            self._bind_router(port, thread_idx, 0, graph_len)

    def _bind_router(self, port, thread_idx, requester_id, graph_len):
        """Serve tensor data requests on a ROUTER socket."""
        ctx = zmq.Context.instance()
        router = ctx.socket(zmq.ROUTER)
        router.setsockopt(zmq.ROUTER_HANDOVER, 1)  # accept re-connections, update identity
        bind_addr = f"tcp://{self.local_ip}:{port}"
        try:
            router.bind(bind_addr)
        except zmq.error.ZMQError:
            # Fall back to all-interfaces if specific IP bind fails
            router.bind(f"tcp://*:{port}")

        print(f"[{self.local_ip}] Serving on port {port} for device {requester_id}")

        _hb = 0
        while self._running:
            _hb += 1
            if _hb % 10 == 0:
                print(f"[{self.local_ip}] _bind_router port={port} still waiting ({_hb}s)...")
            if router.poll(1000):
                frames = router.recv_multipart()
                identity = frames[0]
                request  = frames[1] if len(frames) > 1 else b""
                sample_id = int(frames[2].decode()) if len(frames) > 2 else 0
                print(f"[{self.local_ip}] _bind_router port={port} got frames={len(frames)} req={request}")

                # Android sends "Request Data" (uppercase D); match it case-insensitively
                if request.lower() == b"request data":
                    event = self._data_ready.setdefault(sample_id, threading.Event())
                    latest_identity = identity  # track freshest ZMQ routing identity

                    # Wait for ONNX to finish. Keep updating latest_identity so we always
                    # have the most recently seen routing ID (ROUTER_HANDOVER ensures
                    # routing table is updated whenever Android's DEALER reconnects).
                    # If the relay pipe dies and the tensor never arrives, the event never
                    # fires — detect this and send EOS (token 2) to unblock Android.
                    DEADLOCK_TIMEOUT_S = 90
                    total_wait_s = 0
                    timed_out = False
                    while not event.wait(timeout=1):
                        total_wait_s += 1
                        while router.poll(0):
                            nf = router.recv_multipart()
                            nr = nf[1] if len(nf) > 1 else b""
                            ns = int(nf[2].decode()) if len(nf) > 2 else 0
                            if nr.lower() == b"request data" and ns == sample_id:
                                latest_identity = nf[0]
                        if total_wait_s >= DEADLOCK_TIMEOUT_S:
                            print(f"[{self.local_ip}] DEADLOCK: token event stalled {DEADLOCK_TIMEOUT_S}s "
                                  f"— relay pipe likely dead. Sending EOS to unblock Android.")
                            import struct as _struct
                            time.sleep(0.1)
                            router.send_multipart([
                                latest_identity,
                                sample_id.to_bytes(4, "little"),
                                _struct.pack('<i', 2),  # EOS = token 2 in LLaMA vocab
                            ])
                            event.clear()
                            timed_out = True
                            break

                    if timed_out:
                        continue  # skip normal response; next iteration waits for fresh request

                    # ONNX done. Drain all stale queued messages, still updating identity
                    # to the absolute latest connection seen.
                    drained = 0
                    while router.poll(0):
                        nf = router.recv_multipart()
                        nr = nf[1] if len(nf) > 1 else b""
                        ns = int(nf[2].decode()) if len(nf) > 2 else 0
                        if nr.lower() == b"request data" and ns == sample_id:
                            latest_identity = nf[0]
                        drained += 1
                    if drained:
                        print(f"[{self.local_ip}] Drained {drained} stale queued messages after ONNX")

                    raw_bytes = self.output_data.get(sample_id, bytes(16))

                    # For tailer returning results to header in generation mode:
                    # Android calls deserializeInt(res) expecting a 4-byte little-endian int32
                    # (the predicted next-token ID).  Convert logits → argmax token_id.
                    if self.is_tailer and requester_id == 0 and self.task_type == "generation":
                        import numpy as np, struct
                        arr = np.frombuffer(raw_bytes, dtype=np.float32)
                        vocab_size = 32000
                        seq_len = self._seq_len.get(sample_id, 0)
                        total_elems = len(arr)
                        if seq_len > 0 and total_elems >= seq_len * vocab_size:
                            # Pick logits at the last real token position (right-padded model)
                            arr_last = arr[(seq_len - 1) * vocab_size : seq_len * vocab_size]
                        else:
                            # Fallback: last vocab_size elements
                            arr_last = arr[-vocab_size:] if total_elems >= vocab_size else arr
                        token_id = int(np.argmax(arr_last))
                        print(f"[{self.local_ip}] logit pos={seq_len-1} stats: min={arr_last.min():.3f} max={arr_last.max():.3f} top5={np.argsort(arr_last)[-5:][::-1].tolist()}")
                        response_bytes = struct.pack('<i', token_id)
                        print(f"[{self.local_ip}] Serving token_id={token_id} to header")
                    else:
                        response_bytes = raw_bytes

                    # Android's obtainResultsFromTailer sends ONE "Request Data" then
                    # blocks on recv(0) forever — no retry.  We must send to latest_identity.
                    # Sleep 1.5s so that if the relay dropped the connection during
                    # ONNX, ZMQ DEALER has time to auto-reconnect and ROUTER_HANDOVER
                    # can refresh its routing-table entry to the new connection.
                    time.sleep(1.5)
                    router.send_multipart([
                        latest_identity,
                        sample_id.to_bytes(4, "little"),
                        response_bytes,
                    ])
                    print(f"[{self.local_ip}] Delivered response to latest identity")

                    # Clear the event so the next token step's wait() blocks correctly.
                    event.clear()

    def _resolve_ip(self, ip):
        """Apply ADB/port-forward IP remapping. Returns (host, port_override).
        ip_map values may be 'IP' or 'IP:PORT' — the latter overrides the computed port."""
        mapped = self.ip_map.get(ip, ip)
        if ":" in mapped:
            host, port_str = mapped.rsplit(":", 1)
            try:
                return host, int(port_str)
            except ValueError:
                pass
        return mapped, None

    def _pull_tensor(self, from_device_id, thread_idx, graph_len, sample_id, timeout=60000):
        """Connect DEALER to previous device's ROUTER and request tensor."""
        ctx = zmq.Context.instance()
        dealer = ctx.socket(zmq.DEALER)

        port = BASE_PORT + graph_len * thread_idx + (self.device_id - from_device_id)
        target_ip, port_override = self._resolve_ip(self.ip_graph[from_device_id])
        if port_override is not None:
            port = port_override
        dealer.connect(f"tcp://{target_ip}:{port}")

        dealer.send_multipart([b"Request Data"])

        if dealer.poll(timeout):
            frames = dealer.recv_multipart()
            # frames: [device_id_bytes, tensor_bytes, ..., b'Over']
            dealer.close()
            return frames[1] if len(frames) >= 2 else b""
        dealer.close()
        print(f"[{self.local_ip}] WARNING: Timeout waiting for tensor from device {from_device_id}")
        return b""

    # -----------------------------------------------------------------------
    # Header inference loop
    # -----------------------------------------------------------------------

    def _header_inference_loop(self, graph_len):
        n = len(self.ip_graph)

        # Start prompt input thread
        prompt_thread = threading.Thread(target=self._prompt_input_loop, daemon=True)
        prompt_thread.start()

        print(f"\n[{self.local_ip}] === Header ready ===")
        print(f"[{self.local_ip}] Type a prompt and press Enter to run inference.")
        print(f"[{self.local_ip}] Type 'quit' to end the session.\n")

        for sample_id in range(self.num_sample):
            # Wait for a user prompt
            while not self._prompt_queue and self._running:
                time.sleep(0.1)
            if not self._running:
                break

            with self._prompt_lock:
                prompt = self._prompt_queue.pop(0)
            if prompt.strip().lower() == "quit":
                break

            print(f"[{self.local_ip}] Running inference for: '{prompt}'")
            output_bytes = self._run_onnx_header(prompt)

            # Store output so the next device can pull it
            event = self._data_ready.setdefault(sample_id, threading.Event())
            self.output_data[sample_id] = output_bytes
            event.set()

            if n == 1:
                # No other devices — header is also tailer
                self._print_result(output_bytes)
            else:
                # Wait for final result from tailer
                result = self._pull_tensor(n - 1, 0, graph_len, sample_id)
                self._print_result(result)

    def _run_onnx_header(self, prompt):
        """Tokenize prompt and run the first ONNX shard."""
        import numpy as np
        if not self.sessions:
            print(f"[{self.local_ip}] No sessions — returning random tensor.")
            return np.random.randn(1, 1, 4096).astype(np.float32).tobytes()

        sess = self.sessions[0]
        inp  = sess.get_inputs()[0]

        if self.tokenizer:
            formatted = (
                "<|system|>\nYou are a helpful assistant.</s>\n"
                f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
            )
            tokens = self.tokenizer(formatted, return_tensors="np")
            input_arr = tokens["input_ids"].astype(np.int64)
        else:
            input_arr = np.array(
                [[ord(c) % 32000 for c in prompt[:32]]], dtype=np.int64)

        try:
            outputs = sess.run(None, {inp.name: input_arr})
            result = outputs[0].astype(np.float32)
        except Exception as e:
            print(f"[{self.local_ip}] ONNX error: {e}")
            result = np.random.randn(1, 1, 4096).astype(np.float32)

        print(f"[{self.local_ip}] Header ONNX output shape: {result.shape}")
        return result.tobytes()

    def _prompt_input_loop(self):
        """Background thread: reads prompts from stdin."""
        while self._running:
            try:
                prompt = input(f"[{self.local_ip} PROMPT]> ")
                with self._prompt_lock:
                    self._prompt_queue.append(prompt)
            except EOFError:
                break

    def _print_result(self, tensor_bytes):
        """Decode final output tensor and print the result."""
        import numpy as np
        try:
            arr = np.frombuffer(tensor_bytes, dtype=np.float32)
            token_id = int(np.argmax(arr[-4096:] if len(arr) >= 4096 else arr))
            if self.tokenizer:
                text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                print(f"\n[{self.local_ip}] Response token: '{text}'\n")
            else:
                print(f"\n[{self.local_ip}] Response token ID: {token_id}\n")
        except Exception as e:
            print(f"[{self.local_ip}] Could not decode output: {e}")

    # -----------------------------------------------------------------------
    # Worker / tailer inference loop
    # -----------------------------------------------------------------------

    def _worker_inference_loop(self, graph_len):
        import numpy as np
        from_device_id = self.device_id - 1
        print(f"[{self.local_ip}] Worker ready — waiting for tensor pushes from device {from_device_id}.")

        ctx = zmq.Context.instance()
        # Bind a PULL socket that Android PUSHes tensors to directly (Android→VM direction).
        # Port = BASE_PORT + (device_id - from_device_id) + 4.
        # This avoids the relay:12348→ADB:12347→Android:12346 path which fails for step 1+
        # because the Android emulator's ADB TCP tunnel breaks after the first large transfer.
        # Android→VM works reliably (same direction as obtainResultsFromTailer).
        pull_port = BASE_PORT + (self.device_id - from_device_id) + 4  # e.g. 12350 (graph_len unused here)
        pull_sock = ctx.socket(zmq.PULL)
        pull_sock.setsockopt(zmq.LINGER, 0)
        pull_sock.bind(f"tcp://0.0.0.0:{pull_port}")
        print(f"[{self.local_ip}] PULL socket bound at 0.0.0.0:{pull_port} — Android PUSHes tensors here")

        # For generation tasks Android uses the same sample_id for every token step of
        # a given prompt.
        if self.task_type == "generation" and self.max_length > 0:
            total_steps = self.num_sample * self.max_length
            steps_per_sample = self.max_length
        else:
            total_steps = self.num_sample
            steps_per_sample = 1

        try:
            for step in range(total_steps):
                sample_id = step // steps_per_sample
                print(f"[{self.local_ip}] Waiting for tensor push for step {step}...")

                # Poll in 30s chunks so we get heartbeat logs if Android is slow
                waited = 0
                while not pull_sock.poll(30_000):
                    waited += 30
                    if waited >= 600:
                        print(f"[{self.local_ip}] FATAL: No tensor after 10 min for step {step}. Aborting.")
                        break
                    print(f"[{self.local_ip}] Still waiting for step {step} tensor ({waited}s)...")
                else:
                    waited = None  # poll succeeded
                if waited is not None and waited >= 600:
                    break

                frames = pull_sock.recv_multipart()
                print(f"[{self.local_ip}] Received {len(frames)} frame(s) for step {step}, sizes={[len(f) for f in frames]}")
                # Android sends [id_4bytes, seqLen_4bytes, tensor_bytes]
                if len(frames) >= 3:
                    import struct as _struct
                    seq_len = _struct.unpack('<i', frames[1])[0] if len(frames[1]) == 4 else 0
                    input_bytes = frames[2]
                else:
                    seq_len = 0
                    input_bytes = frames[1] if len(frames) >= 2 else b""
                self._seq_len[sample_id] = seq_len
                print(f"[{self.local_ip}] seq_len={seq_len} for sample {sample_id} step {step}")
                if not input_bytes:
                    print(f"[{self.local_ip}] WARNING: Empty tensor for step {step}. Skipping.")
                    continue

                output_bytes = self._run_onnx_worker(input_bytes, sample_id)

                event = self._data_ready.setdefault(sample_id, threading.Event())
                self.output_data[sample_id] = output_bytes
                event.set()

                if self.is_tailer and self.task_type != "generation":
                    self._print_result(output_bytes)
        finally:
            pull_sock.close()

    @staticmethod
    def _deserialize_tensor_vector(data: bytes):
        """Deserialize the Android SerializeTensorVectorToBytes format.

        Wire format (little-endian, Android AArch64):
          [numTensors: 8 bytes (size_t)]
          For each tensor:
            [tensorType: 4 bytes (ONNXTensorElementDataType enum)]
            [numDimensions: 8 bytes (size_t)]
            [dim_0 .. dim_n: 8 bytes each (int64_t)]
            [raw tensor data]
        """
        import struct
        import numpy as np

        ONNX_DTYPE = {
            1:  np.float32,   # FLOAT
            2:  np.uint8,     # UINT8
            3:  np.int8,      # INT8
            4:  np.uint16,    # UINT16
            5:  np.int16,     # INT16
            6:  np.int32,     # INT32
            7:  np.int64,     # INT64
            9:  np.bool_,     # BOOL
            11: np.float64,   # DOUBLE
            12: np.uint32,    # UINT32
            13: np.uint64,    # UINT64
        }

        offset = 0
        num_tensors = struct.unpack_from('<Q', data, offset)[0]
        offset += 8

        tensors = []
        for _ in range(num_tensors):
            tensor_type = struct.unpack_from('<I', data, offset)[0]
            offset += 4

            num_dims = struct.unpack_from('<Q', data, offset)[0]
            offset += 8

            shape = list(struct.unpack_from(f'<{num_dims}q', data, offset))
            offset += num_dims * 8

            dtype = ONNX_DTYPE.get(tensor_type, np.float32)
            num_elements = 1
            for dim in shape:
                if dim > 0:
                    num_elements *= dim
            arr = np.frombuffer(data, dtype=dtype, count=num_elements, offset=offset)
            arr = arr.reshape([d if d > 0 else 1 for d in shape])
            offset += num_elements * arr.itemsize
            tensors.append(arr)

        return tensors

    def _run_onnx_worker(self, input_bytes, sample_id):
        """Run intermediate/tailer ONNX shard on received tensor bytes."""
        print(f"[{self.local_ip}] Received tensor bytes: {len(input_bytes)}, as float32 count: {len(input_bytes)//4}")

        import numpy as np
        if not self.sessions:
            print(f"[{self.local_ip}] No sessions — passing through tensor.")
            return input_bytes

        sess = self.sessions[0]
        inp_names  = [i.name  for i in sess.get_inputs()]
        inp_shapes = [i.shape for i in sess.get_inputs()]
        print(f"[{self.local_ip}] Module1 expects: {list(zip(inp_names, inp_shapes))}")

        try:
            tensors = self._deserialize_tensor_vector(input_bytes)
            print(f"[{self.local_ip}] Deserialized {len(tensors)} tensor(s) from blob")

            if len(tensors) != len(inp_names):
                print(f"[{self.local_ip}] WARNING: deserialized {len(tensors)} tensors "
                      f"but model needs {len(inp_names)} inputs — padding with zeros")

            ONNX_NP_DTYPE = {
                "tensor(float)":   np.float32,
                "tensor(float16)": np.float16,
                "tensor(double)":  np.float64,
                "tensor(int32)":   np.int32,
                "tensor(int64)":   np.int64,
                "tensor(int8)":    np.int8,
                "tensor(uint8)":   np.uint8,
                "tensor(bool)":    np.bool_,
            }

            feed = {}
            for i, inp in enumerate(sess.get_inputs()):
                expected_dtype = ONNX_NP_DTYPE.get(inp.type, np.float32)
                if i < len(tensors):
                    feed[inp.name] = tensors[i].astype(expected_dtype)
                else:
                    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
                    feed[inp.name] = np.zeros(shape, dtype=expected_dtype)

            t0 = time.time()
            outputs = sess.run(None, feed)
            result  = outputs[0].astype(np.float32)
            print(f"[{self.local_ip}] ONNX inference took {time.time()-t0:.2f}s, output shape: {result.shape}")
        except Exception as e:
            print(f"[{self.local_ip}] ONNX worker error (sample {sample_id}): {e}")
            result = np.frombuffer(input_bytes, dtype=np.float32)

        label = "Tailer" if self.is_tailer else "Worker"
        print(f"[{self.local_ip}] {label} shape={result.shape}")
        return result.tobytes()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_ip_map(values):
    """Parse --ip-map OLD=NEW entries into a dict."""
    result = {}
    for entry in (values or []):
        parts = entry.split("=", 1)
        if len(parts) == 2:
            result[parts[0].strip()] = parts[1].strip()
        else:
            print(f"WARNING: Ignoring malformed --ip-map entry: {entry}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="LinguaLinked Python Device Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Worker node
  python device_client.py --role worker --ip 127.0.0.2 --server 172.17.0.2

  # Header node
  python device_client.py --role header --ip 127.0.0.3 --server 172.17.0.2 --model llama-2-7b

  # With ADB forwarding (Android emulator at 10.0.2.15 mapped to 127.0.0.1)
  python device_client.py --role worker --ip 127.0.0.2 --server 172.17.0.2 \\
      --ip-map 10.0.2.15=127.0.0.1
        """
    )
    parser.add_argument("--role",   choices=["header", "worker"], required=True,
                        help="Device role: 'header' drives inference, 'worker' processes a layer")
    parser.add_argument("--ip",     required=True,
                        help="This device's IP address as seen by other devices")
    parser.add_argument("--server", required=True,
                        help="Root server IP address")
    parser.add_argument("--model",  default=None,
                        help="Model name — required for header (e.g. llama-2-7b)")
    parser.add_argument("--port",   type=int, default=ROOT_PORT,
                        help=f"Root server ZMQ port (default: {ROOT_PORT})")
    parser.add_argument("--ip-map", nargs="*", metavar="OLD=NEW",
                        help="Remap device IPs for ADB forwarding, e.g. 10.0.2.15=127.0.0.1")
    parser.add_argument("--model-dir", default=None,
                        help="Absolute path to pre-extracted model shard directory. "
                             "Overrides the default ./device_models/<ip>/ location.")

    args = parser.parse_args()

    if args.role == "header" and not args.model:
        parser.error("--model is required when --role is header")

    ip_map = parse_ip_map(args.ip_map)

    sim = DeviceSimulator(
        role=args.role,
        local_ip=args.ip,
        server_ip=args.server,
        model_name=args.model,
        server_port=args.port,
        ip_map=ip_map,
        model_dir=args.model_dir,
    )
    sim.run()


if __name__ == "__main__":
    main()
