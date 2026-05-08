install requirements

## Running coordinator node
docker run -it --name lingualinked `
  -p 5000:5000 `
  -v "<path to repo>:/app" `
  -v "<path to model>:/app/llama-2-7b" `
  -w /app `
  python:3.10-slim /bin/bash

cd frameworks/lingualinked/

pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

apt-get update && apt-get install -y patchelf

patchelf --clear-execstack /usr/local/lib/python3.10/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-310-x86_64-linux-gnu.so

docker build -t lingualinked .

docker compose run --rm --service-ports coordinator

python root.py


## Running emulator for header/worker
Install android studio

Create a phone device with 6gb memory

Open ./android/distributed_inference_demo

Run distribute_ui, green play button android studio

## Running thin worker node

See project root for setting up VM instance

sudo apt update && sudo apt install -y python3-pip python3-venv git

sudo apt install openssh-server -y

ssh vboxuser@192.168.1.157 #example ip, run this on your local machine for easy terminal

python3 -m venv ~/ll

source ~/ll/bin/activate

pip install pyzmq onnxruntime numpy psutil

python3 ~/device_client.py --role worker --ip 10.0.2.2 --server 192.168.1.148 --ip-map 10.0.2.15=100.105.155.44:12348 --model-dir device_models/192_168_1_121

## Cloud VM Deployment (True Distributed — Each Node on a Separate VM)

Oracle Cloud's default network security policies block the ZMQ ports LinguaLinked needs,
which is why it fails there. The following providers work without special workarounds.

### Recommended cloud providers

| Provider | Cost/node | Ease of firewall config | Notes |
|---|---|---|---|
| **DigitalOcean Droplets** | $6–12/mo | ★★★★★ | Best choice. VPC private networking within a region is free and fast. |
| **Hetzner Cloud** | €4–7/mo | ★★★★☆ | Cheapest. Good EU + US regions. Simple Firewall UI. |
| **Vultr** | $5–10/mo | ★★★★☆ | Similar to DigitalOcean. |
| **AWS EC2** (t3.small) | ~$15/mo | ★★★☆☆ | Free tier available. Security Groups are straightforward but more steps. |
| **GCP** (e2-small) | ~$13/mo | ★★★☆☆ | Free tier. VPC firewall rules via web console or gcloud CLI. |

### Required open ports

Open these **inbound** TCP rules on every VM's firewall/security-group:

| Port | Who needs it open | Purpose |
|---|---|---|
| 22 | All VMs | SSH |
| 23456 | **Coordinator VM only** | ZMQ ROUTER — device registration |
| 34567 | **Coordinator VM only** | ZMQ ROUTER — hardware monitor |
| 12345–12360 | **All worker/header VMs** | P2P tensor transfer (ROUTER/DEALER) |
| 55555 | **All worker/header VMs** | Bandwidth test server (TCP) |

DigitalOcean example (using their CLI):
```
doctl compute firewall create \
  --name lingualinked \
  --inbound-rules "protocol:tcp,ports:22,address:0.0.0.0/0 \
                   protocol:tcp,ports:23456,address:0.0.0.0/0 \
                   protocol:tcp,ports:34567,address:0.0.0.0/0 \
                   protocol:tcp,ports:12345-12360,address:0.0.0.0/0 \
                   protocol:tcp,ports:55555,address:0.0.0.0/0"
```

### IP addressing on cloud VMs

Each `device_client.py --ip` argument must be the IP **other VMs use to reach this node**:

- **Same cloud region / VPC** — use the VM's **private/internal IP** (e.g. `10.x.x.x`).
  Traffic stays within the provider's network: zero egress cost, lower latency.
- **Different regions or providers** — use the VM's **public IP**.

The `--server` argument is always the coordinator's reachable IP (private or public, same rule).

### Example: 2-node cloud setup (coordinator + 1 worker/header) on DigitalOcean

```bash
# --- Coordinator VM (runs root.py inside Docker) ---
export MODEL_PATH=/opt/models/llama-2-7b   # where weights live on this VM
docker compose run --rm --service-ports coordinator
python root.py

# --- Worker/Header VM (separate Droplet, private IP 10.0.0.5) ---
pip install pyzmq onnxruntime numpy psutil
python3 device_client.py \
    --role header \
    --ip 10.0.0.5 \               # this VM's private IP
    --server 10.0.0.4 \           # coordinator's private IP
    --model llama-2-7b
```

### Running coordinator without Docker on a Linux cloud VM

```bash
cd frameworks/lingualinked/
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python root.py
```
