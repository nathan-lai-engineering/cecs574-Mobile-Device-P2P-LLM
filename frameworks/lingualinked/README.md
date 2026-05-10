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
