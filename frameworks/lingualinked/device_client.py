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
                 server_port=ROOT_PORT, ip_map=None):
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

        self.model_dir = Path(f"./device_models/{local_ip.replace('.', '_')}")
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

        # Server sends back monitor signal: b"True" or b"False"
        monitor_msg = self.root_socket.recv_multipart()
        need_monitor = monitor_msg[0].decode().strip() == "True"
        print(f"[{self.local_ip}] Registered. Monitor: {need_monitor}")

    # -----------------------------------------------------------------------
    # Phase 2: Ready → Open → Prepare → Initialized → Start → Running → Finish
    # -----------------------------------------------------------------------

    def _lifecycle(self):
        # Ready
        print(f"[{self.local_ip}] Sending Ready...")
        self.root_socket.send_multipart([b"Ready", self.local_ip.encode()])

        # Open — server sends configuration
        msg = self.root_socket.recv_multipart()
        if msg[0] != b"Open":
            print(f"[{self.local_ip}] ERROR: Expected Open, got {msg[0]}")
            return
        self._parse_open_config(msg)

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

        self.device_id  = self.ip_graph.index(self.local_ip)
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
        chunks = []

        while True:
            chunk_msg = self.root_socket.recv_multipart()
            chunk = chunk_msg[0]
            if chunk == b"":
                break
            chunks.append(chunk)
            total = sum(len(c) for c in chunks)
            print(f"[{self.local_ip}]   {total:,} bytes received...", end="\r")

        data = b"".join(chunks)
        print(f"\n[{self.local_ip}] Download complete: {len(data):,} bytes")

        with open(zip_path, "wb") as f:
            f.write(data)

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

        onnx_files = sorted(self.model_dir.rglob("*.onnx"))
        if not onnx_files:
            print(f"[{self.local_ip}] WARNING: No .onnx files in {self.model_dir}. Passthrough mode.")
            return

        for path in onnx_files:
            print(f"[{self.local_ip}] Loading {path.name}...")
            sess = ort.InferenceSession(str(path))
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
        bind_addr = f"tcp://{self.local_ip}:{port}"
        try:
            router.bind(bind_addr)
        except zmq.error.ZMQError:
            # Fall back to all-interfaces if specific IP bind fails
            router.bind(f"tcp://*:{port}")

        print(f"[{self.local_ip}] Serving on port {port} for device {requester_id}")

        while self._running:
            if router.poll(1000):
                frames = router.recv_multipart()
                identity = frames[0]
                request  = frames[1] if len(frames) > 1 else b""
                sample_id = int(frames[2].decode()) if len(frames) > 2 else 0

                if request == b"Request data":
                    event = self._data_ready.setdefault(sample_id, threading.Event())
                    event.wait(timeout=60)
                    tensor_bytes = self.output_data.get(sample_id, bytes(16))
                    router.send_multipart([
                        identity,
                        self.device_id.to_bytes(4, "little"),
                        tensor_bytes,
                        b"Over"
                    ])

    def _resolve_ip(self, ip):
        """Apply ADB/port-forward IP remapping if configured."""
        return self.ip_map.get(ip, ip)

    def _pull_tensor(self, from_device_id, thread_idx, graph_len, sample_id, timeout=60000):
        """Connect DEALER to previous device's ROUTER and request tensor."""
        ctx = zmq.Context.instance()
        dealer = ctx.socket(zmq.DEALER)

        port = BASE_PORT + graph_len * thread_idx + (self.device_id - from_device_id)
        target_ip = self._resolve_ip(self.ip_graph[from_device_id])
        dealer.connect(f"tcp://{target_ip}:{port}")

        dealer.send_multipart([b"Request data", str(sample_id).encode()])

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
            tokens = self.tokenizer(prompt, return_tensors="np")
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
        print(f"[{self.local_ip}] Worker ready — waiting for data from device {self.device_id - 1}.")

        for sample_id in range(self.num_sample):
            # Pull activation tensor from previous device
            input_bytes = self._pull_tensor(
                self.device_id - 1, 0, graph_len, sample_id)

            output_bytes = self._run_onnx_worker(input_bytes, sample_id)

            event = self._data_ready.setdefault(sample_id, threading.Event())
            self.output_data[sample_id] = output_bytes
            event.set()

            if self.is_tailer:
                self._print_result(output_bytes)

    def _run_onnx_worker(self, input_bytes, sample_id):
        """Run intermediate/tailer ONNX shard on received tensor bytes."""
        import numpy as np
        if not self.sessions:
            print(f"[{self.local_ip}] No sessions — passing through tensor.")
            return input_bytes

        sess = self.sessions[0]
        inp  = sess.get_inputs()[0]

        try:
            arr = np.frombuffer(input_bytes, dtype=np.float32)
            # Reshape to match ONNX model's expected input shape
            expected_shape = [d if d is not None and d > 0 else 1
                              for d in inp.shape]
            arr = arr[:int(np.prod(expected_shape))].reshape(expected_shape)
            outputs = sess.run(None, {inp.name: arr})
            result  = outputs[0].astype(np.float32)
        except Exception as e:
            print(f"[{self.local_ip}] ONNX worker error (sample {sample_id}): {e}")
            result = np.frombuffer(input_bytes, dtype=np.float32)

        label = "Tailer" if self.is_tailer else "Worker"
        print(f"[{self.local_ip}] {label} ONNX output shape: {result.shape}")
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
    )
    sim.run()


if __name__ == "__main__":
    main()
