# LinguaLinked

Distributed LLM inference across Android devices and Python worker nodes. The coordinator splits the model and orchestrates inference; the header node drives token generation; worker nodes each run one model shard.

> **On first run**, the coordinator takes several minutes to analyze and shard TinyLlama (2.2 GB) — it runs multiple full ONNX forward passes to profile the model. Subsequent runs use the cached shards and skip this step.

**Startup order: Coordinator → Workers → Header (APK last)**

---

## Prerequisites

Install all of the following before starting:

| Tool | Required by | How to install |
|---|---|---|
| **Docker Desktop** (Windows) | Coordinator | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) — enable WSL 2 backend during setup |
| **Android Studio** (any recent version) | Header APK | [developer.android.com/studio](https://developer.android.com/studio) — install via the full installer, not the command-line tools only |
| **ADB in system PATH** | Port forwarding | Bundled with Android Studio. Add `%LOCALAPPDATA%\Android\Sdk\platform-tools` to your Windows PATH environment variable |
| **Tailscale** (optional, required for remote workers) | Cross-machine networking | [tailscale.com/download](https://tailscale.com/download) — sign in with your Tailscale account after installing |

Verify Docker is running and ADB is accessible before proceeding:

```powershell
docker version
adb version
```

Both commands must succeed without errors.

---

## Repository Layout

All paths below are written relative to the repository root. On this machine:
- Repository root: `c:\CSULB\cecs574-Mobile-Device-P2P-LLM\`
- LinguaLinked root: `c:\CSULB\cecs574-Mobile-Device-P2P-LLM\frameworks\lingualinked\`

The two directories you work in most:

```
<repo-root>/frameworks/lingualinked/           ← coordinator, device_client.py, Docker files
<repo-root>/frameworks/lingualinked/android/   ← Android Studio project and APK source
```

---

## Coordinator

Runs on your local machine via Docker. Handles device registration, model sharding, and shard distribution to all connected workers.

### Build the Docker image (first time only)

Open PowerShell and navigate to the lingualinked directory:

```powershell
cd c:\CSULB\cecs574-Mobile-Device-P2P-LLM\frameworks\lingualinked
docker build -t lingualinked-coordinator .
```

The build downloads PyTorch (CPU-only, ~200 MB) and all coordinator dependencies. It finishes in 5–10 minutes on first run. Subsequent builds reuse cached layers and finish in seconds.

If the build fails due to a GPU reservation error, edit `docker-compose.yaml` (in the same `frameworks/lingualinked/` directory) and remove the `deploy:` block entirely:

```yaml
# Remove these lines if you have no NVIDIA GPU:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: all
#           capabilities: [gpu]
```

### Start the coordinator

From the `frameworks/lingualinked/` directory (the same one containing `docker-compose.yaml`):

```powershell
cd c:\CSULB\cecs574-Mobile-Device-P2P-LLM\frameworks\lingualinked
docker compose -f docker-compose.yaml run --rm --service-ports coordinator
```

This opens a bash shell inside the container. The project directory is mounted at `/app` inside the container — you are already at `/app` when the shell opens. Run:

```bash
python root.py
```

Wait until the terminal prints `start listening`. Leave this terminal open and running for the entire session.

> TinyLlama (~2.2 GB) downloads automatically from HuggingFace Hub on first run. Subsequent runs detect the cached shards and skip re-downloading.

---

## Header — Android Studio Emulator

The header runs as the APK on an Android emulator on your local machine. Start this **after** all workers have registered with the coordinator.

### Step 1 — Configure `config.properties` before building

This file controls which coordinator the app connects to and what IP it registers itself as. Edit it before every build.

Full path:
```
c:\CSULB\cecs574-Mobile-Device-P2P-LLM\frameworks\lingualinked\android\distributed_inference_demo\distribute_ui\src\main\assets\config.properties
```

Set the contents to:

```properties
# Coordinator address.
# 10.0.2.2 is the Android emulator's fixed gateway to the host machine — use this when
# the coordinator (root.py) is running on the same machine as the emulator.
server_ip=10.0.2.2

# This device's own IP, registered with the coordinator so other nodes can connect back.
# Leave blank when all nodes (coordinator, workers, emulator) are on the same machine.
# Set to the host machine's Tailscale IP (e.g. 100.x.x.1) when workers are on Hetzner.
device_ip=
```

To find your local machine's Tailscale IP (if using Hetzner workers):

```powershell
tailscale ip -4
```

> This file is compiled into the APK at build time. Any change to `config.properties` requires a full rebuild in Android Studio and a reinstall on the emulator.

### Step 2 — Open the Android project in Android Studio

1. Launch Android Studio
2. Click **Open** (or **File → Open** if a project is already open)
3. Navigate to and select the directory:
   ```
   c:\CSULB\cecs574-Mobile-Device-P2P-LLM\frameworks\lingualinked\android\distributed_inference_demo
   ```
4. Click **OK** and wait for Gradle sync to complete — this can take several minutes on first open

### Step 3 — Create an AVD (first time only)

1. Open **Device Manager** (`View → Tool Windows → Device Manager` or the phone icon in the right toolbar)
2. Click **Create Virtual Device**
3. Select **Phone → Pixel 6** (or any phone category device), click **Next**
4. Select system image **API 34 (Android 14, x86_64)** — click **Download** next to it if not already installed, then select it and click **Next**
5. Under **AVD Name**, give it a recognizable name (e.g. `LinguaLinked_Header`)
6. Click **Show Advanced Settings** and set:
   - **RAM**: `6144` MB (6 GB minimum — the model shard requires significant memory)
   - **VM heap**: `1024` MB
7. Click **Finish**

### Step 4 — Run the APK on the emulator

1. Select your new AVD in the device dropdown at the top toolbar of Android Studio
2. In the **Project** panel on the left, ensure the `distribute_ui` module is selected
3. Click the green **Run ▶** button (or press **Shift+F10**)
4. Android Studio builds the APK, installs it on the emulator, and launches the app automatically

### Step 5 — Configure role in the app

Once the app opens on the emulator:
1. Select role: **Header**
2. Select model: **TinyLlama**
3. Tap **Next**

The app registers with the coordinator. Watch the coordinator terminal — it should print the registration.

### Step 6 — ADB port forwarding (required when workers are on a remote machine)

Workers on Hetzner or any other remote machine connect to the header's ZMQ ports. These ports live inside the emulator, but the workers see only the local machine's Tailscale IP. ADB port forwarding bridges the gap: it forwards each port on the local machine's network interface into the emulator.

Run the following in a **new** PowerShell window (keep the coordinator shell open):

```powershell
12345..12360 | ForEach-Object { adb forward tcp:$_ tcp:$_ }
adb forward tcp:55555 tcp:55555
```

Run this after the emulator is fully booted (home screen visible) and before workers register with the coordinator.

To verify the forwards are active:

```powershell
adb forward --list
```

You should see entries for ports 12345–12360 and 55555.

---

## Worker — Local VM

For a Python worker running inside a VirtualBox or VMware VM on the same machine as the coordinator.

### Initial VM setup

Inside the VM terminal:

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git
```

Optionally SSH in from the host for a more comfortable terminal:

```bash
# Run on the host machine — replace <vm-ip> with the VM's IP (found via `ip addr show` inside VM)
ssh vboxuser@<vm-ip>
```

### Clone the repository

```bash
git clone https://github.com/YOUR_FORK/cecs574-Mobile-Device-P2P-LLM.git
cd cecs574-Mobile-Device-P2P-LLM/frameworks/lingualinked
```

### Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pyzmq onnxruntime numpy psutil
```

### Run the worker

```bash
cd ~/cecs574-Mobile-Device-P2P-LLM/frameworks/lingualinked
source .venv/bin/activate

python3 device_client.py \
    --role worker \
    --ip <vm-ip> \       # this VM's IP — run `ip addr show` inside the VM to find it
    --server <host-ip>   # host machine's LAN IP — run `ipconfig` on Windows to find it
```

`<host-ip>` is the IP of the Windows machine running the coordinator Docker container, on the same LAN or bridge network the VM uses.

---

## Worker — Hetzner (Cloud)

See [hetzner_cloud_setup.md](hetzner_cloud_setup.md) for the complete setup guide — Hetzner CAX11 ARM64 nodes connected via Tailscale.

### Quick reference

**Install Tailscale on each worker:**

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
tailscale ip -4   # note this 100.x.x.x address — used as --ip below
```

**Install Python 3.10 and dependencies (Ubuntu 22.04):**

```bash
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-pip

python3.10 -m venv .venv && source .venv/bin/activate

pip install pyzmq==25.1.0 onnxruntime==1.16.1 numpy==1.26.4 psutil==5.9.5 \
    transformers==4.33.3 tokenizers sentencepiece protobuf
```

**Clone the repo and run worker:**

```bash
git clone https://github.com/YOUR_FORK/cecs574-Mobile-Device-P2P-LLM.git
cd cecs574-Mobile-Device-P2P-LLM/frameworks/lingualinked
source .venv/bin/activate

python device_client.py \
    --role worker \
    --ip 100.x.x.2 \      # this worker's own Tailscale IP (from `tailscale ip -4`)
    --server 100.x.x.1    # coordinator machine's Tailscale IP
```

Set `device_ip=<host Tailscale IP>` in `config.properties` (full path above) before building the APK.

---

## Known Misunderstandings

1. **Workers must register before the header.** The coordinator stops accepting registrations `TIMEOUT` seconds after the last device connects (`TIMEOUT = 10` seconds, at the top of `frameworks/lingualinked/root.py`). If the header connects before all workers, the optimizer sees fewer devices than intended and generates fewer shards. Start all workers first, wait for each one to print `waiting for assignment`, then launch the APK.

2. **`config.properties` must be set before building the APK.** The file at `frameworks/lingualinked/android/distributed_inference_demo/distribute_ui/src/main/assets/config.properties` is baked into the APK at compile time. Changing it after the APK is installed has no effect — you must rebuild and reinstall via Android Studio.

3. **`--ip` must be the IP reachable by all other nodes.** Each `device_client.py` invocation must pass its own IP (`--ip`) as the address other nodes will use to connect back to it. On Hetzner, this is the Tailscale `100.x.x.x` address. Passing a loopback or wrong interface address causes inference to hang.

4. **Model caching is auto-detected.** The coordinator checks whether ONNX shards already exist for the current device count (`cached_ok` in `root.py`). On first run the shards are always generated and distributed (~2–5 minutes for TinyLlama). If you change the number of workers, delete `frameworks/lingualinked/onnx_model/to_send/` so shards are regenerated for the new device count.

5. **Extra workers stand by automatically.** If more workers connect than the optimizer needs for the current model/device count, the extras receive a `Close` signal and wait 30 seconds before re-registering. They will be picked up in the next session automatically — no manual restart needed.

---

## Reproducing the Paper Experiment

This section describes how to replicate the test environment and results from the paper *"A Performance Evaluation of P2P Inference Frameworks: LinguaLinked vs. Exo on Mobile-like devices using Virtual Machines"*. The paper measures **Time to First Token (TTFT)** across N=1, N=2, and N=3 worker nodes for both LinguaLinked (TinyLlama) and EXO (Qwen3.5 0.7b 8bit).

### Reference results (Table 1 from the paper)

Full results for both frameworks are in the root [README.md](../../README.md). LinguaLinked results:

| Metric | N=1 | N=2 | N=3 |
|---|---|---|---|
| Min (s) | 20.1 | 38.7 | 39.6 |
| Average (s) | 21.7 | 39.2 | 39.6 |
| Max (s) | 23.9 | 39.6 | 41.2 |

> TinyLlama's architecture limits the optimizer to 2 independent subgraphs. At N=3, the third worker receives no shard and stands by — TTFT at N=3 is identical to N=2.

---

### Environment overview

| Component | Paper setup |
|---|---|
| Host machine | i7-12700K, 64 GB DDR4 4000 MT/s |
| Worker VMs | Debian 12, VirtualBox, x86-64, 1–4 vCPU, 4 GB RAM |
| Cloud workers (alternative) | Hetzner CX23 (x86-64, 2 vCPU, 4 GB RAM) |
| Header | Android Studio emulator, 6 GB RAM |
| Model | TinyLlama 1.1B |
| Network conditions tested | 1000 Mbps, 100 Mbps, 10 Mbps (VirtualBox virtual switch) |
| CPU conditions tested | 1–4 vCPUs, execution cap varied per VM |

---

### LinguaLinked experiment (VirtualBox local VMs)

**Before starting:** Set up the base Debian 12 VM and clone it per the [root README Base VM Setup](../../README.md#base-vm-setup-shared-by-both-frameworks). The clones should be named `ll-worker-1`, `ll-worker-2`, `ll-worker-3` and all attached to the `lingualinked` NAT Network. Find each VM's IP via `ip addr show` (look for the `10.0.3.x` address) — you'll need it for every worker command below.

#### Step 1 — Install LinguaLinked worker dependencies on each VM

SSH into each VM from the host. VirtualBox NAT Network does not expose VMs to the host directly; add a second adapter in host-only mode for SSH, or use the VirtualBox console. Inside the VM:

```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git

git clone https://github.com/YOUR_FORK/cecs574-Mobile-Device-P2P-LLM.git
cd cecs574-Mobile-Device-P2P-LLM/frameworks/lingualinked

python3 -m venv .venv
source .venv/bin/activate
pip install pyzmq==25.1.0 onnxruntime==1.16.1 numpy==1.26.4 psutil==5.9.5 \
    transformers==4.33.3 tokenizers sentencepiece protobuf
```

#### Step 2 — Run a LinguaLinked TTFT test (baseline, N=2)

This is the minimum viable test — 1 coordinator, 2 workers, 1 header emulator. All at 1000 Mbps.

**Terminal 1 — Coordinator (host machine, Docker):**

```powershell
cd c:\CSULB\cecs574-Mobile-Device-P2P-LLM\frameworks\lingualinked
docker compose -f docker-compose.yaml run --rm --service-ports coordinator
# Inside the container:
python root.py
```

Wait for `start listening`.

**Terminal 2 — Worker 1 (ll-worker-1 VM):**

```bash
cd ~/cecs574-Mobile-Device-P2P-LLM/frameworks/lingualinked
source .venv/bin/activate
python3 device_client.py \
    --role worker \
    --ip <ll-worker-1-ip> \     # 10.0.3.x from `ip addr show`
    --server <host-ip>          # host machine's IP on the NAT network gateway (10.0.3.1 typically)
```

**Terminal 3 — Worker 2 (ll-worker-2 VM):**

```bash
cd ~/cecs574-Mobile-Device-P2P-LLM/frameworks/lingualinked
source .venv/bin/activate
python3 device_client.py \
    --role worker \
    --ip <ll-worker-2-ip> \
    --server <host-ip>
```

**Android Studio — Header (after both workers have registered):**

1. Run the APK on the Android emulator (Shift+F10 in Android Studio)
2. In the app: select **Header**, **TinyLlama**, tap **Next**
3. Wait for the app to show the chat screen (model shard received)
4. Type a prompt (e.g., `"Explain what a neural network is."`) and submit

**Reading TTFT:** After the response appears in the app, the TTFT is displayed in the app's metrics panel. Alternatively, stream logcat from PowerShell and filter for timing output:

```powershell
adb logcat -s Lingual_backend | Select-String "TTFT\|Computation Time\|Results"
```

#### Step 3 — Run all N configurations

Repeat step 2 for each N value:

| Configuration | Workers to start | Expected avg TTFT |
|---|---|---|
| N=1 | Start only `ll-worker-1` | ~21.7 s |
| N=2 | Start both `ll-worker-1` and `ll-worker-2` | ~39.2 s |
| N=3 | Start all three workers | ~39.6 s (same as N=2 — TinyLlama caps at 2 shards) |

Before each N configuration, delete the shard cache so the coordinator re-runs the optimizer for the new device count:

```bash
# Inside the Docker container (coordinator terminal):
rm -rf /app/onnx_model/to_send/
```

Then restart `python root.py` and repeat the registration sequence.

Collect a minimum of 3 TTFT measurements per configuration and record min/average/max.

#### Step 4 — Simulate node join/leave mid-run

To simulate a worker dropping out during inference, simply close that VM's terminal or suspend the VM while inference is running:

```powershell
VBoxManage controlvm "ll-worker-2" pause     # pause the VM mid-inference
VBoxManage controlvm "ll-worker-2" resume    # resume after observing the effect
```

For the EXO experiment, see [frameworks/exo/README.md](../exo/README.md). The VMs created above (Debian 12 x86-64, same NAT Network) are reused — EXO requires no additional VM configuration beyond installing its own Python dependencies on each VM.
