Team 2
2peers
Nathan Lai
Kevin Kongwattanachai

---

# Summary
This repo stores several other repos for ease including: LinguaLinked, EXO, and Llama.cpp. 
1. https://github.com/zjc664656505/LinguaLinked-Inference
2. https://github.com/exo-explore/exo
3. https://github.com/ggml-org/llama.cpp

We have edited and modified some repos to allow for better and virtual testing. Currently the testing is centered around locally run VM's, but we would like to expand instructions and modifications to allow running on cloud VM's to allow for true networked P2P.

The focus of our research is to compare the repo's and determine viability of each framework. Discovering the practical strengths and weaknesses of each framework is beneficial as well for future research.

## Experiment Results
By running each framework on consistent specs and VM images, we were able to determine EXO had better inference speed scaling though Lingualinked had better overall TTFT through the use of its coordinator server.

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
You will be setting up the virtual VM's below to run each framework.

The next step will be setting up each framework on the instances as well as configuring your local machine's network settings to allow communication between emulators and VM's.

## Setup VM

### Setup debian

Download and Install Virtualbox

Download debian 12 iso

Create new virtualbox VM from debian ISO (disk size at least 50gb)

Once debian finished installing from ISO, shut down VM

### Create virtual switch

Go to main network tab, create new NAT network

### Edit VM settings

Right Click on the vm and select settings

System: 1 cpu core, 6gb ram

Network: select the Nat network you created, randomize mac address

### Setting up specific frameworks

Follow instructions on specific framework readme

Clone vm instance after framework is fully installed with randomized Nat network to create identical instances.



