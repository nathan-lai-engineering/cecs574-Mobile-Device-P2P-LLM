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

pip install pyzmq onnxruntime numpy

python3 ~/device_client.py --role worker --ip 10.0.2.2 --server 192.168.1.148 --ip-map 10.0.2.15=100.105.155.44:12348 --model-dir device_models/192_168_1_121
