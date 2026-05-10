# LinguaLinked on Hetzner Cloud — Deployment Guide

This guide covers deploying LinguaLinked worker nodes on Hetzner Cloud, enabling true distributed inference across cloud machines connected via Tailscale. The coordinator and header (Android emulator) continue to run on your local Windows machine.

**Startup order: Coordinator (local) → Workers (Hetzner) → Header APK (local, last)**

---

## 1. Hetzner Node Recommendations

All worker nodes use **CAX11** (2 vCPU ARM64, 4 GB RAM, ~€3.79/month). Use the same region (e.g., `nbg1`) so nodes share Hetzner's private backbone.

| Node Role | Hetzner Type | Architecture | RAM | Notes |
|---|---|---|---|---|
| Worker | CAX11 (2 vCPU, 4 GB) | ARM64 | 4 GB | Holds 1 model shard; no GPU needed |

Start with TinyLlama for a first test run — each worker needs ~1–1.5 GB RAM for its shard, leaving headroom on 4 GB nodes.

---

## 2. Tailscale Setup

Tailscale handles all cross-network connectivity between your local machine and Hetzner nodes. No manual port forwarding or VPN configuration is needed.

### Step 1 — Install Tailscale on your local Windows machine (coordinator + header)

Download and install from [tailscale.com/download](https://tailscale.com/download). Sign in to your Tailscale account during setup. After installation, Tailscale runs in the system tray.

### Step 2 — Note Tailscale IPs for all nodes

After all devices are authenticated, find each device's Tailscale IP in the [Tailscale admin console](https://login.tailscale.com/admin/machines) or by running:

```bash
tailscale ip -4   # run on each node after authenticating
```

Example assignment (your actual IPs will differ — the `100.x.x.x` prefix is fixed but the last octets are assigned by Tailscale):

```
local machine (coordinator + header):   100.x.x.1
hetzner-worker-1:                        100.x.x.2
hetzner-worker-2:                        100.x.x.3
```

Write down all three IPs — you will need them in step 4 (startup sequence).

---

## 3. Hetzner Server Setup

### Create servers

Go to **Servers → Add Server** in the Hetzner Cloud console ([console.hetzner.cloud](https://console.hetzner.cloud)) and configure each worker node as follows. Repeat for each worker node.

**Location**

Pick one region and use it for all nodes (e.g., Nuremberg `nbg1` or Falkenstein `fsn1`). Intra-region traffic between Hetzner nodes is faster and free.

**Image**

Select **Ubuntu 22.04** — required. The Python 3.10 dependency steps below are written for Ubuntu 22.04. If you accidentally select 24.04, see the note in step 2 of the per-node setup below.

**Type**

Click the **Arm** tab (not x86) and select **CAX11** (2 vCPU, 4 GB RAM, ~€3.79/mo).

**Networking**

| Setting | Value | Reason |
|---|---|---|
| **Public IPv4** | Enabled (on) | Required for initial SSH access and Tailscale auth |
| **Public IPv6** | Disabled | Not needed |
| **Private network** | Disabled | Tailscale replaces the need for Hetzner private networking |

**Additional options**

| Setting | Value |
|---|---|
| **Placement group** | None |
| **Backups** | Disabled |
| **Volumes** | None |
| **Firewall** | `lingualinked-fw` (create this in advance — see below) |
| **SSH key** | Your public SSH key |
| **Name** | e.g. `ll-worker-1`, `ll-worker-2` |

**Creating the firewall** (one time, before creating servers):

Go to **Firewalls → Create Firewall** and add these inbound rules:

| Protocol | Port | Source | Purpose |
|---|---|---|---|
| TCP | 22 | Any | SSH access |
| TCP | 12345–12360 | Any | ZMQ inter-device tensor ports |
| TCP | 55555 | Any | ZMQ fallback port |
| UDP | 41641 | Any | Tailscale WireGuard |

After creation, note the server's **Public IPv4** from the server detail page. This is only used for the initial SSH login and Tailscale authentication — all LinguaLinked traffic flows over Tailscale afterward.

### SSH into each node

From your local machine (PowerShell or any terminal):

```bash
ssh root@<public-ipv4>
# Example:
ssh root@5.75.1.10
```

If your SSH key is not in the default location (`~/.ssh/id_rsa` or `~/.ssh/id_ed25519`):

```bash
ssh -i ~/.ssh/your_key root@<public-ipv4>
```

The first login shows a host key fingerprint prompt — type `yes` to accept. You land as `root` in the home directory (`/root`).

---

### Per-node software setup

Run the following blocks in order on each worker node. All commands are run as `root` via SSH.

**Step 1 — Verify OS version**

```bash
lsb_release -a
```

Expected output:
```
Distributor ID: Ubuntu
Release:        22.04
```

If this shows 24.04, either recreate the server with the correct image, or install Python 3.10 via the deadsnakes PPA before continuing:

```bash
# Only needed on Ubuntu 24.04 — skip entirely on 22.04
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

The command prints a URL like `https://login.tailscale.com/a/xxxxxxxx`. Open that URL in your browser and authenticate the node to your Tailscale account. Once authenticated, the terminal on the Hetzner node returns to the prompt.

Then note the node's Tailscale IP — this is the `--ip` value you will pass to `device_client.py`:

```bash
tailscale ip -4
# Example output: 100.x.x.2
```

**Step 4 — Clone the repository**

```bash
git clone https://github.com/YOUR_FORK/cecs574-Mobile-Device-P2P-LLM.git
```

This creates the directory `/root/cecs574-Mobile-Device-P2P-LLM/`. The LinguaLinked code is inside it:

```
/root/cecs574-Mobile-Device-P2P-LLM/frameworks/lingualinked/
```

Navigate there now — all remaining commands run from this directory:

```bash
cd /root/cecs574-Mobile-Device-P2P-LLM/frameworks/lingualinked
```

**Step 5 — Create a Python virtual environment**

```bash
python3.10 -m venv .venv
source .venv/bin/activate

# Upgrade pip inside the venv
pip install --upgrade pip
```

Verify the venv is active — your prompt should show `(.venv)` as a prefix.

**Step 6 — Install Python dependencies**

Workers only run ONNX inference — PyTorch is not needed:

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

If this fails with a missing library error, install the OpenMP runtime:

```bash
sudo apt-get install -y libgomp1
```

---

## 4. Startup Sequence

**Order matters — follow this exactly.**

### A. Start the coordinator (local machine, first)

The coordinator (`root.py`) runs on your local machine. It can run either natively (if you have a Python venv set up) or via Docker (see `README.md` for the Docker path — it is the recommended approach on Windows).

**Via Docker (recommended on Windows):**

Open PowerShell and run:

```powershell
cd c:\CSULB\cecs574-Mobile-Device-P2P-LLM\frameworks\lingualinked
docker compose -f docker-compose.yaml run --rm --service-ports coordinator
```

Inside the container shell that opens:

```bash
python root.py
```

**Via native Python venv (alternative):**

```powershell
cd c:\CSULB\cecs574-Mobile-Device-P2P-LLM\frameworks\lingualinked
.\.venv\Scripts\activate     # Windows venv activation
python root.py
```

In both cases, wait until the terminal prints:

```
start listening
```

Leave the coordinator running — do not close this terminal.

### B. Start workers (each Hetzner node, second)

SSH into each Hetzner worker. Then:

```bash
cd /root/cecs574-Mobile-Device-P2P-LLM/frameworks/lingualinked
source .venv/bin/activate

python device_client.py \
    --role worker \
    --ip 100.x.x.2 \       # THIS worker's own Tailscale IP (from `tailscale ip -4`)
    --server 100.x.x.1     # coordinator's Tailscale IP (your local machine)
```

Replace `100.x.x.2` with this specific node's Tailscale IP and `100.x.x.1` with your local machine's Tailscale IP.

The worker prints its registration status:

```
[100.x.x.2] ========== LinguaLinked Device Simulator ==========
[100.x.x.2] Role: worker | Server: 100.x.x.1:23456
[100.x.x.2] Registering with coordinator...
```

The coordinator terminal prints each registration as it arrives. Start all workers before the 10-second timeout (`TIMEOUT = 10` at the top of `frameworks/lingualinked/root.py`). If the timeout fires before all workers connect, the optimizer runs with fewer devices than intended.

If you have more workers than the optimizer needs for the current model, the extras receive a `Close` signal and automatically retry in 30 seconds — no manual restart needed.

### C. Set up ADB port forwarding (local machine, before starting the header)

Workers on Hetzner connect to the header's ZMQ ports using the header's registered IP (`device_ip` in `config.properties`, which is your local machine's Tailscale IP). ADB port forwarding redirects traffic arriving at those ports on the local machine into the Android emulator.

Open a **new** PowerShell window and run:

```powershell
12345..12360 | ForEach-Object { adb forward tcp:$_ tcp:$_ }
adb forward tcp:55555 tcp:55555
```

Verify the forwards are active:

```powershell
adb forward --list
```

You should see two lines per port — one for each direction.

### D. Start the header — APK on Android Studio Emulator (local machine, last)

1. Confirm `config.properties` is set correctly before building. The file is at:
   ```
   c:\CSULB\cecs574-Mobile-Device-P2P-LLM\frameworks\lingualinked\android\distributed_inference_demo\distribute_ui\src\main\assets\config.properties
   ```
   It should contain:
   ```properties
   server_ip=10.0.2.2
   device_ip=100.x.x.1    # your local machine's Tailscale IP
   ```

2. Open Android Studio, build, and run the `distribute_ui` app on the emulator (Shift+F10).

3. In the app: select role **Header**, select model **TinyLlama**, tap **Next**.

The header's registration signals the coordinator to run the optimizer and begin model distribution to each worker. The first run is slow — the coordinator profiles the model and then sends each shard (several hundred MB) over Tailscale to each worker. Watch the coordinator terminal for progress. Subsequent runs reuse cached shards and skip the transfer.

---

## 5. Known Misunderstandings

1. **`--ip` must be the Tailscale IP of that specific node.** Each `device_client.py` invocation must pass its own `100.x.x.x` Tailscale IP via `--ip`, not the public IP or `127.0.0.1`. The coordinator and header use this address to open connections back to the worker. Passing the wrong IP causes inference to hang silently.

2. **`--server` is the coordinator's Tailscale IP, not the Hetzner node's public IP.** The coordinator runs on your local Windows machine. Its Tailscale IP (e.g., `100.x.x.1`) is what workers must reach on port `23456`.

3. **Model caching is auto-detected on subsequent runs.** The coordinator checks whether ONNX shards already exist for the current device count (`cached_ok` in `root.py`). On first run the shards are always generated and sent. If you change the number of workers, delete the `frameworks/lingualinked/onnx_model/to_send/` directory (on the local machine inside the Docker container, or in the native venv directory) so shards are regenerated for the new count.

4. **Workers must register before the header.** The coordinator stops accepting registrations `TIMEOUT` seconds after the last device connects (`TIMEOUT = 10` at the top of `frameworks/lingualinked/root.py`). Start all workers first, wait for each to print its registration message in the coordinator terminal, then launch the APK.

5. **`config.properties` is baked into the APK at build time.** Changing `device_ip` or `server_ip` after the APK is installed has no effect. Every change requires a rebuild in Android Studio (`Build → Rebuild Project`) and a reinstall on the emulator.
