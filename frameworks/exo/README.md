# EXO

True P2P distributed LLM inference. Each node discovers peers via UDP broadcast and arranges itself in a ring topology. EXO uses tensor parallelism — every node handles a portion of every layer simultaneously — in addition to pipeline parallelism.

> **Architecture note:** EXO is designed for physically co-located devices on high-speed connections (Thunderbolt/USB4). Its tensor parallelism is highly sensitive to network latency, which limits its effectiveness over slower or wireless links.

> **Platform note:** EXO's MLX dependency targets Apple Silicon. x86-64 Linux support (via PyTorch backend) works but is not the primary focus. AArch64 Linux (ARM VMs) is **not supported** due to MLX incompatibility — use x86-64 VMs only.

---

## Reproducing the Paper Experiment (EXO)

This section covers the EXO side of the results in *"A Performance Evaluation of P2P Inference Frameworks: LinguaLinked vs. Exo on Mobile-like devices using Virtual Machines"*. The paper measured **Time to First Token (TTFT)** for EXO running **Qwen3.5 0.7b 8bit** across N=1, N=2, and N=3 worker nodes.

> EXO is **incompatible** with TinyLlama and Llama2. The paper substituted Qwen3.5 0.7b 8bit due to its similar size and confirmed compatibility with EXO v1.0.68.

### Reference results (from paper Table 1)

| Metric | N=1 | N=2 | N=3 |
|---|---|---|---|
| Min (s) | 66.0 | 51.0 | 49.0 |
| Average (s) | 68.4 | 52.2 | 52.4 |
| Max (s) | 70.0 | 55.0 | 55.0 |

> **First-prompt rule:** Discard the result of the first prompt in every run. EXO performs UDP broadcast node discovery on the first request, causing TTFT 250–350% higher than subsequent prompts. Collect measurements starting from the second prompt of each run.

---

### Prerequisites

| Requirement | Value |
|---|---|
| VM architecture | x86-64 only (AArch64 not supported) |
| OS | Debian 12 (Ubuntu 22.04 also works) |
| Python | 3.11 or later |
| EXO version | 1.0.68 (as used in paper) |
| Model | `qwen3:0.7b-q8_0` |
| RAM per VM | 4 GB minimum |
| vCPUs per VM | 1–4 (varied for heterogeneity testing) |

VMs must be on the same broadcast domain (VirtualBox NAT Network) so EXO's UDP discovery works. See the root [README.md](../../README.md) for VirtualBox VM creation, NAT Network setup, and bandwidth/CPU throttling commands.

---

### Step 1 — Install Python 3.11 on each VM

Debian 12 ships Python 3.11. Confirm it is installed:

```bash
python3.11 --version
# Expected: Python 3.11.x
```

If missing:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip
```

---

### Step 2 — Install EXO on each VM

```bash
python3.11 -m venv ~/exo-venv
source ~/exo-venv/bin/activate
pip install --upgrade pip
pip install exo-explore==1.0.68
```

If v1.0.68 is unavailable from PyPI, install from the git tag:

```bash
pip install "git+https://github.com/exo-explore/exo.git@v1.0.68"
```

Verify the install:

```bash
exo --version
```

---

### Step 3 — Start EXO on each VM

SSH into each VM and start EXO. Start all VMs for a given N configuration before submitting any prompts.

```bash
source ~/exo-venv/bin/activate
exo
```

EXO starts a discovery daemon and a local API server. Wait for it to print cluster membership — it discovers other running EXO nodes on the same NAT Network automatically via UDP broadcast.

To confirm all nodes are visible, open the EXO web UI from any VM's browser (or via a port-forwarded connection from the host):

```
http://<vm-ip>:8080
```

The web UI lists all discovered nodes and shows how the model is partitioned across them.

---

### Step 4 — Warm up the model (first time only)

EXO downloads the model on first use and distributes shards automatically. Run a warmup prompt on one VM to pre-download before timing:

```bash
source ~/exo-venv/bin/activate

curl -s http://127.0.0.1:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:0.7b-q8_0","messages":[{"role":"user","content":"warmup"}]}'
```

Wait for the response before running any timed tests.

---

### Step 5 — Measure TTFT

TTFT is the time from when the prompt request is sent to when the first token appears in the streamed response. The script below measures this precisely and discards the first prompt automatically.

Run this from inside any one VM (EXO routes the request across all nodes):

```bash
source ~/exo-venv/bin/activate

python3 - <<'EOF'
import requests, time, json

PROMPT = "Explain what a neural network is."
URL = "http://127.0.0.1:52415/v1/chat/completions"
BODY = {
    "model": "qwen3:0.7b-q8_0",
    "messages": [{"role": "user", "content": PROMPT}],
    "stream": True
}

def measure_ttft():
    t0 = time.time()
    with requests.post(URL, json=BODY, stream=True) as r:
        for line in r.iter_lines():
            if line and line.startswith(b"data: ") and line != b"data: [DONE]":
                data = json.loads(line[6:])
                if data.get("choices", [{}])[0].get("delta", {}).get("content"):
                    return time.time() - t0
    return None

print("Warmup (discarded)...")
measure_ttft()

results = []
for i in range(3):
    ttft = measure_ttft()
    results.append(ttft)
    print(f"Run {i+1}: TTFT = {ttft:.2f}s")

print(f"\nMin: {min(results):.2f}s  Avg: {sum(results)/len(results):.2f}s  Max: {max(results):.2f}s")
EOF
```

Run this script for each N configuration and record the min/average/max TTFT.

---

### Step 6 — Run all N configurations

Stop EXO on all VMs between configurations (`pkill -f exo`), then restart with the target number of VMs.

| Configuration | VMs running EXO | Expected avg TTFT |
|---|---|---|
| N=1 | 1 VM only | ~68.4 s |
| N=2 | 2 VMs | ~52.2 s |
| N=3 | 3 VMs | ~52.4 s |

The model download happens once and is cached per VM. After the first run, subsequent N configurations do not re-download.

To stop EXO cleanly on a VM:

```bash
pkill -f "exo"
```

---

### Network and CPU simulation

The same VirtualBox `VBoxManage` commands used for LinguaLinked apply here. See the root [README.md](../../README.md) for the full commands. Quick reference:

```powershell
# Change bandwidth limit (host PowerShell, VM can be running):
VBoxManage bandwidthctl "exo-vm-1" set "NetLimit" --limit 100m   # 100 Mbps
VBoxManage bandwidthctl "exo-vm-1" set "NetLimit" --limit 10m    # 10 Mbps

# Change CPU execution cap (live — simulates throttling):
VBoxManage controlvm "exo-vm-1" cpuexecutioncap 50   # 50% (throttled)
VBoxManage controlvm "exo-vm-1" cpuexecutioncap 100  # full speed

# Change vCPU count (VM must be stopped):
VBoxManage modifyvm "exo-vm-1" --cpus 2
```

---

For the LinguaLinked experiment, see [frameworks/lingualinked/README.md](../lingualinked/README.md).
