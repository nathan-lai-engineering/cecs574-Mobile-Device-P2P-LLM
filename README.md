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
By running each framework on consistent hardware specifications and VM images, we observed distinct performance profiles:
* **EXO:** Demonstrated superior **Inference Speed Scaling**. Its architecture is highly effective at increasing throughput as more nodes are added to the cluster. Architecture analysis reveals typically higher max device count as well.
* **LinguaLinked:** Provided a superior **Time to First Token (TTFT)** generally. The centralized coordinator server efficiently manages the workload distribution, reducing the "startup" latency of a generation. 

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
You will set up multiple Debian-based Virtual Machines to simulate a networked cluster. This setup allows for testing P2P communication within a controlled local virtual switch before expanding to cloud-based VM testing. Cloud-based VM testing is not currently supported, but will be built out over time.

## Setup VM
Setting up an individual worker device through a virtual machine.

### Setup debian
Setup the initial Debian image, and ensure it is able to run on your local machine

1. Download and Install Virtualbox

2. Download debian 12 ISO

3. Create new virtualbox VM from debian ISO (disk size at least 50gb)

4. Once debian finished installing from ISO, shut down VM

### Create virtual switch
Ensuring networking is enabled without conflict.

1. Go to main network tab, create new NAT network

### Edit VM settings
Editing configuration to allow consistent specs.

1. Right Click on the vm and select settings

2. System: 1 cpu core, 6gb ram

3. Network: select the Nat network you created, randomize mac address

### Setting up specific frameworks
Next instructions are depedent on framework, please see README on each framework root folder to see instructions to set up that framework.

1. Clone initial instance to configure for specific framework. Keep the base image, so that only clones are used for each new device. I recommend each instance only contain one of the frameworks to prevent depedency conflicts and consistency.

2. Follow instructions on specific framework readme

3. Clone vm instance after framework is fully installed with randomized Nat network to create identical instances.




