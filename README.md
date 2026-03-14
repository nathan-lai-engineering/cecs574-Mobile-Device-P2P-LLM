Team 2
2peers
Nathan Lai
Kevin Kongwattanachai

---

## Project Structure
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

# Setup VM

## Setup debian

Download and Install Virtualbox

Download debian 12 iso

Create new virtualbox VM from debian ISO (disk size at least 50gb)

Once debian finished installing from ISO, shut down VM

## Create virtual switch

Go to main network tab, create new NAT network

## Edit VM settings

Right Click on the vm and select settings

System: 1 cpu core, 6gb ram

Network: select the Nat network you created, randomize mac address

## Setting up specific frameworks

Follow instructions on specific framework readme

Clone vm instance after framework is fully installed with randomized Nat network to create identical instances.



