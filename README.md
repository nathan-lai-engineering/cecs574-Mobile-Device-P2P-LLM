## Team 2: 2peers
**Collaborators:** Nathan Lai & Kevin Kongwattanachai  
**Project:** A Performance Evaluation of P2P Inference Frameworks: LinguaLinked vs. Exo on Mobile-like devices using Virtual Machines

---

# Summary
This repository serves as a centralized hub for researching and benchmarking distributed LLM inference frameworks. It contains modified forks of leading P2P and hybrid inference projects, optimized for local virtualized testing environments.
1. https://github.com/zjc664656505/LinguaLinked-Inference
2. https://github.com/exo-explore/exo
3. https://github.com/ggml-org/llama.cpp

We have edited and modified some repos to allow for better and virtual testing. Currently the testing is centered around locally run VM's, but we would like to expand instructions and modifications to allow running on cloud VM's to allow for true networked P2P.

The focus of our research is to compare the repo's and determine viability of each framework. Discovering the practical strengths and weaknesses of each framework is beneficial as well for future research.

## Experiment Results

Metric: **Time to First Token (TTFT)** across N=1, N=2, N=3 worker nodes. Lower is better.

| Metric | LL N=1 | LL N=2 | LL N=3 | EXO N=1 | EXO N=2 | EXO N=3 |
|---|---|---|---|---|---|---|
| Min (s) | 20.1 | 38.7 | 39.6 | 66.0 | 51.0 | 49.0 |
| Average (s) | 21.7 | 39.2 | 39.6 | 68.4 | 52.2 | 52.4 |
| Max (s) | 23.9 | 39.6 | 41.2 | 70.0 | 55.0 | 55.0 |

- **LinguaLinked** (TinyLlama 1.1B): Lower baseline TTFT. Hits a ceiling at N=2 because TinyLlama's independent subgraph count limits sharding to 2 nodes — N=3 produces identical results.
- **EXO** (Qwen3.5 0.7b 8bit): Higher baseline TTFT but improves steadily with N. Tensor parallelism is sensitive to network latency, which limits gains at N=3.

## Architecture Differences
### LinguaLinked
Hybrid P2P with a mobile-first focus. Utilizes a coordinator server to divide inference work between worker devices. Inference model is still run in serial using Pipeline Parallelism, just divided among devices, therefore benefits through larger prompts with multiple tokens. Maximum device count for a given inference model is dependent on modularity of the used model, as it calculates max worker count by the amount of logically indepdent subgraphs of the model.
### EXO
True P2P focused around physically connected Apple devices. Utilizes true parallelism with tensor paralleism powered largely by extremely low overhead with high transfer speed thunderbolt data cables. Models are partitioned by hidden layer counts which determines the max worker count.
### Llama.cpp
Standalone, there is no P2P here. Used as a baseline to determine whether P2P Frameworks can outperform single device inference.

# Project Structure
Project is organized into following folders

```text
.
├── frameworks/            # forks of the original repos
│   ├── exo/                    
│   │   └── README.md
│   ├── lingualinked/            # root folder has coordinator and thin worker nodes
│   │   ├── README.md
│   │   ├── root.py                     # coordinator node
│   │   ├── device_client.py            # thin worker python script
│   │   └── android/
│   │       └── distributed_inference_demo/       # android app of the worker/header node
│   └── llama/
│       └── README.md
└── README.md
```

# Instructions for Reproduction of Results

## Overview

You will set up multiple Debian 12 x86-64 Virtual Machines in VirtualBox to simulate a networked cluster. Each VM represents one worker node. Both frameworks share the same VM base image — clone it after OS install and configure each clone for its specific framework.

Framework-specific reproduction instructions are in each framework's README:

- **LinguaLinked** (coordinator + Android header + Python workers): [frameworks/lingualinked/README.md](frameworks/lingualinked/README.md)
- **EXO** (true P2P, tensor parallelism): [frameworks/exo/README.md](frameworks/exo/README.md)

For cloud-based worker nodes (Hetzner), see [frameworks/lingualinked/hetzner_cloud_setup.md](frameworks/lingualinked/hetzner_cloud_setup.md).

---

## Base VM Setup (shared by both frameworks)

### Step 1 — Install VirtualBox

Download and install VirtualBox from [virtualbox.org/wiki/Downloads](https://www.virtualbox.org/wiki/Downloads). Add `VBoxManage` to your system PATH:

```
C:\Program Files\Oracle\VirtualBox\   ← add this to Windows PATH
```

### Step 2 — Create a NAT Network

In VirtualBox: **File → Tools → Network Manager → NAT Networks → Create**. Set:
- Name: `lingualinked`
- IPv4 Prefix: `10.0.3.0/24`
- Enable DHCP: checked

All worker VMs attach to this network so they can reach each other and the host.

### Step 3 — Create the base Debian 12 VM

1. In VirtualBox: **New**
2. Set:
   - Name: `ll-base`
   - ISO Image: Debian 12 netinst amd64 ISO from [debian.org/download](https://www.debian.org/download)
   - Type: Linux, Version: Debian (64-bit)
3. Hardware: **4096 MB RAM**, **4 CPUs**
4. Hard disk: **50 GB** (dynamically allocated — enough for model shards)
5. After creation: **Settings → Network → Adapter 1** → NAT Network → `lingualinked`
6. Install Debian 12 minimal (no desktop environment needed). Create a regular user and note the password.

After install, find the VM's IP from inside it — you'll need this for every framework command:

```bash
ip addr show      # look for the 10.0.3.x address on eth0 or enp0s3
```

### Step 4 — Clone the base VM for each worker

Keep `ll-base` untouched. Right-click → **Clone** for each worker node needed:
- Name each clone: `ll-worker-1`, `ll-worker-2`, `ll-worker-3`
- Select **Full Clone**
- Under **MAC Address Policy**: select **Generate new MAC addresses for all network adapters**

Each clone boots with its own IP on the NAT Network.

### Step 5 — Set up bandwidth throttling per VM (for network condition tests)

Run from PowerShell on the **host machine**. The VM must be **stopped** before adding the NIC group assignment.

```powershell
# Create bandwidth group (1000 Mbps default) and assign to NIC
VBoxManage bandwidthctl "ll-worker-1" add "NetLimit" --type network --limit 1000m
VBoxManage modifyvm "ll-worker-1" --nicbandwidthgroup1 "NetLimit"
```

Repeat for each VM. To change the limit while the VM is **running**:

```powershell
VBoxManage bandwidthctl "ll-worker-1" set "NetLimit" --limit 100m   # 100 Mbps
VBoxManage bandwidthctl "ll-worker-1" set "NetLimit" --limit 10m    # 10 Mbps
VBoxManage bandwidthctl "ll-worker-1" set "NetLimit" --limit 1000m  # restore
```

### Step 6 — Set CPU allocation per VM (for heterogeneity tests)

To change vCPU count (VM must be **stopped**):

```powershell
VBoxManage modifyvm "ll-worker-1" --cpus 2    # 2 vCPUs
VBoxManage modifyvm "ll-worker-1" --cpus 4    # restore
```

To change CPU execution cap (VM can be **running** — simulates thermal throttling):

```powershell
VBoxManage controlvm "ll-worker-1" cpuexecutioncap 50    # 50% speed
VBoxManage controlvm "ll-worker-1" cpuexecutioncap 100   # full speed
```

### Step 7 — Install framework-specific software on each clone

Follow the framework README for the remaining steps:

- **LinguaLinked workers**: [frameworks/lingualinked/README.md → Reproducing the Paper Experiment](frameworks/lingualinked/README.md#reproducing-the-paper-experiment)
- **EXO workers**: [frameworks/exo/README.md](frameworks/exo/README.md)




