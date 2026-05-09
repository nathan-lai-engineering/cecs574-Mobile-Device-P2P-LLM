# LinguaLinked

Distributed LLM inference across Android devices and Python worker nodes. The coordinator splits the model and orchestrates inference; the header node drives token generation; worker nodes each run one model shard.

> On first run, the coordinator takes several minutes to analyze and shard TinyLlama (2.2 GB) — it runs multiple full ONNX forward passes to profile the model. Subsequent runs use the cached shards and skip this step.

**Startup order: Coordinator → Workers → Header (APK last)**

---

## Coordinator

Runs on your local machine via Docker. Handles device registration, model sharding, and shard distribution.

### Build the image (first time only)

```bash
cd frameworks/lingualinked
docker build -t lingualinked-coordinator .
```

### Start coordinator

```bash
docker compose run --rm --service-ports coordinator
```

This opens a shell inside the container at `/app`. Then run:

```bash
python root.py
```

Wait until `start listening` appears before starting any workers.

> TinyLlama downloads automatically from HuggingFace Hub on first run. Subsequent runs use the local shard cache and skip re-downloading.

---

## Header — Android Studio Emulator

Runs the APK on your local machine. Start this **after** all workers are registered with the coordinator.

### Configure `config.properties` before building

Edit `android/distributed_inference_demo/distribute_ui/src/main/assets/config.properties`:

```properties
# Coordinator address — 10.0.2.2 is the emulator's gateway to the host machine
server_ip=10.0.2.2

# Leave blank for local-only setup.
# Set to your Tailscale IP if workers are on remote machines (Hetzner).
device_ip=
```

> This file is baked into the APK at build time. Any change requires a rebuild and reinstall.

### Run the APK

1. Open `android/distributed_inference_demo` in Android Studio
2. Create a phone device with at least 6 GB RAM in AVD Manager
3. Run `distribute_ui` (green play button)
4. In the app: select role **Header**, select model **TinyLlama**, tap **Next**

### ADB port forwarding

Required when workers are on a different machine (VM or Hetzner) so they can connect back to the emulator's ZMQ ports:

```powershell
# Run in PowerShell on the local machine
12345..12360 | ForEach-Object { adb forward tcp:$_ tcp:$_ }
adb forward tcp:55555 tcp:55555
```

---

## Worker — Local VM

For a worker running in a VirtualBox or VMware VM on your local machine.

### Initial setup

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git openssh-server
```

SSH into the VM from the host for an easier terminal (optional):

```bash
ssh vboxuser@<vm-ip>
```

### Install dependencies

```bash
python3 -m venv ~/ll
source ~/ll/bin/activate
pip install pyzmq onnxruntime numpy psutil
```

### Run worker

```bash
python3 device_client.py \
    --role worker \
    --ip <vm-ip> \      # this VM's IP, reachable by the coordinator and header
    --server <host-ip>  # coordinator machine's LAN IP
```

---

## Worker — Hetzner (Cloud)

See [HETZNER_DEPLOYMENT.md](HETZNER_DEPLOYMENT.md) for the complete setup guide — Hetzner CAX11 ARM64 nodes connected via Tailscale.

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

**Run worker:**

```bash
python device_client.py \
    --role worker \
    --ip 100.x.x.2 \    # this worker's Tailscale IP
    --server 100.x.x.1  # coordinator's Tailscale IP (local machine)
```

Set `device_ip=<host Tailscale IP>` in `config.properties` before building the APK.
