# LinguaLinked on Hetzner Cloud — Deployment Notes

This is only applicable for the Lingualinked worker nodes, allowing cloud execution to simulate true distributed computing.

---

## 1. Hetzner Node Recommendations

All worker nodes use **CAX11** (2 vCPU ARM64, 4 GB RAM, ~€3.79/month). Use the same region (e.g., `nbg1`) so nodes share Hetzner's private backbone.

| Node Role | Hetzner Type | Architecture | RAM | Notes |
|---|---|---|---|---|
| Worker | CAX11 (2 vCPU, 4 GB) | ARM64 | 4 GB | Holds 1 model shard; no execstack fix needed |

**Model memory guidance per shard (with 2 workers):**
- BLOOM-560M: ~0.3 GB/shard — CAX11 works easily
- TinyLlama (1.1B): ~0.6 GB/shard — CAX11 works
- LLaMA-2-7B: ~3.5 GB/shard — CAX11 is borderline; requires HuggingFace approval

Start with BLOOM-560M or TinyLlama for a first test run.

---

## 2. Tailscale Setup

Tailscale handles all cross-network connectivity. No manual VPN config or port forwarding needed on your local machine or Hetzner.

### Step 1 — Install Tailscale everywhere

**On your local Windows machine running coordinator and header:**
Download and install from [tailscale.com/download](https://tailscale.com/download). Sign in to your Tailscale account.

**On each Hetzner worker:**
Covered in Section 3 (Per-Node Software Setup, Step 2) — Tailscale installation is part of the per-node setup sequence.

**Android Studio emulator:**
Tailscale runs on the Windows host — no separate install needed for the emulator. The emulator reaches the host via `10.0.2.2`. Set `device_ip=<host Tailscale IP>` in `config.properties` so the APK registers a routable address that Hetzner workers can connect back to.

### Step 2 — Note Tailscale IPs

After all devices are authenticated, check IPs in the [Tailscale admin console](https://login.tailscale.com/admin/machines) or via:
```bash
tailscale ip -4   # on each node
```

Example assignment:
```
local machine:    100.x.x.1   (coordinator + header emulator)
hetzner-worker-1: 100.x.x.2
hetzner-worker-2: 100.x.x.3
```

---

## 3. Hetzner Server Setup

### Create Servers

Go to **Servers → Add Server** and configure each worker node as follows:

**Location**
- Pick one region and use it for all nodes (e.g., Nuremberg `nbg1` or Falkenstein `fsn1`). Intra-region traffic between Hetzner nodes is faster and free.

**Image**
- **Ubuntu 22.04** — required. Selecting the wrong image will break the Python install steps below.

**Type**
- Select **Arm** tab → **CAX11** (2 vCPU, 4 GB RAM, ~€3.79/mo)

**Networking**

| Setting | Value | Reason |
|---|---|---|
| **Public IPv4** | **Enabled (on)** | Required for access using SSH |
| **Public IPv6** | Disabled | Not needed |
| **Private network** | Disabled | Tailscale will serve as private network |

**Additional options**

| Setting | Value |
|---|---|
| **Placement group** | None |
| **Backups** | Disabled |
| **Volumes** | None |
| **Firewall** | `lingualinked-fw` |
| **SSH key** | Your key |
| **Name** | e.g. `ll-worker-1`, `ll-worker-2` |

After creation, the **public IPv4** appears on the server detail page. Note it for the initial SSH + Tailscale setup. After that, all LinguaLinked traffic uses the Tailscale `100.x.x.x` address.

### SSH Into Each Node

```bash
ssh root@<public-ipv4>
# Example:
ssh root@5.75.1.10
```

If your SSH key is not in the default location:
```bash
ssh -i ~/.ssh/your_key root@<public-ipv4>
```

First login will show a host key fingerprint prompt, just type `yes` to accept. You land as `root`.

### Per-Node Software Setup

SSH into each worker and run the following blocks in order.

**Step 1 — Verify OS version**

```bash
lsb_release -a
# Distributor ID: Ubuntu
# Release: 22.04   <-- must be 22.04, not 24.04
```

If this shows 24.04, recreate the server with the correct Ubuntu 22.04 image. Alternatively, install Python 3.10 via the deadsnakes PPA:
```bash
# Only if on Ubuntu 24.04 — skip this block on 22.04
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
```

**Step 2 — Install system prerequisites**

Python 3.10 is not pre-installed on the Hetzner Ubuntu 22.04 minimal image. Install it along with all required build tools:

```bash
sudo apt-get update
sudo apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-distutils \
    python3-pip \
    curl \
    git \
    build-essential \
    cmake
```

**Step 3 — Install Tailscale**

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

Follow the auth URL printed in the terminal to authenticate this node to your Tailscale account. After authentication:
```bash
tailscale ip -4
# Note this 100.x.x.x address — it is the --ip value for device_client.py
```

**Step 4 — Clone the repo**

```bash
git clone https://github.com/YOUR_FORK/cecs574-Mobile-Device-P2P-LLM.git
cd cecs574-Mobile-Device-P2P-LLM/frameworks/lingualinked
```

**Step 5 — Create a Python virtual environment**

```bash
python3.10 -m venv .venv
source .venv/bin/activate

# Upgrade pip inside the venv
pip install --upgrade pip
```

**Step 6 — Install Python dependencies**

Workers only run ONNX inference — torch is not needed:

```bash
pip install \
    pyzmq==25.1.0 \
    onnxruntime==1.16.1 \
    numpy==1.26.4 \
    psutil==5.9.5 \
    transformers==4.33.3 \
    tokenizers \
    sentencepiece \
    protobuf
```

Verify ONNX Runtime installed correctly:
```bash
python3.10 -c "import onnxruntime; print(onnxruntime.__version__)"
# Expected: 1.16.1
```

---

## 4. Startup Sequence

**Order matters — start in this order.**

### A. Start Coordinator (local machine, first)

```bash
# In the lingualinked directory on your local machine
source .venv/bin/activate   # or activate your local venv

# Only needed if using llama-2-7b:
# export LLAMA2_7B_PATH=/path/to/hf_models/llama-2-7b

python root.py
```

Wait until you see `start listening` printed. Leave it running.

### B. Start Workers (Hetzner nodes, second)

On each Hetzner worker (replace IPs with your actual Tailscale IPs):

```bash
cd cecs574-Mobile-Device-P2P-LLM/frameworks/lingualinked
source .venv/bin/activate

python device_client.py \
    --role worker \
    --ip 100.x.x.2 \        # THIS worker's Tailscale IP
    --server 100.x.x.1      # Local machine's Tailscale IP (coordinator)
```

Each worker registers with the coordinator. Coordinator prints each registration.

### C. Start Header — APK on Android Studio Emulator (local machine, last)

1. Set `config.properties` before building:
   ```properties
   server_ip=10.0.2.2         # emulator gateway to host
   device_ip=100.x.x.1        # host machine's Tailscale IP
   ```

2. Build and run the APK from Android Studio. Select Header role in the app.

3. Set up ADB port forwarding so Hetzner workers can reach the emulator's ZMQ ports:
   ```powershell
   # Run in PowerShell on the local machine
   12345..12360 | ForEach-Object { adb forward tcp:$_ tcp:$_ }
   adb forward tcp:55555 tcp:55555
   ```

The header's registration signals the coordinator to begin optimization and model distribution.

---

## 5. Port Reference

| Port | Protocol | Direction | Purpose |
|---|---|---|---|
| 23456 | TCP | Devices → Coordinator | Registration + model file transfer |
| 34567 | TCP | Devices → Coordinator | Hardware monitoring |
| 12345-12360 | TCP | Node → Node | Tensor passing between nodes |
| 55555 | TCP | Node → Node | Bandwidth measurement |

All traffic flows over the Tailscale network (`100.x.x.x`). No manual firewall rules needed.

---

## 6. Known Misunderstandings

1. **`--ip` must be the Tailscale IP of that node.** Each `device_client.py` invocation must pass its own `100.x.x.x` Tailscale IP so the coordinator and other nodes can connect back to it.

2. **`MODEL_EXIST_ON_DEVICE = False` on first run.** If set to `True`, the coordinator skips sending the shard and workers crash with a missing model error.

3. **Workers must register before the header.** The coordinator stops accepting registrations `TIMEOUT` seconds after the last device connects. If the header connects before all workers, the optimizer will see fewer devices than intended.

4. **APK `config.properties` must be set before building.** The file is baked into the APK at build time. Changing it after installation requires a rebuild and reinstall.
