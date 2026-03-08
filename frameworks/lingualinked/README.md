requirements

for coordinator node (docker)
docker run -it --name lingualink_node_1 `
  -p 5000:5000 `
  -v "<path to repo>:/app" `
  -v "<path to model>:/app/llama-2-7b" `
  -w /app `
  python:3.10-slim /bin/bash
cd frameworks/lingualinked/
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python root.py --port 5000

for header/worker node (emulator)
android studio
rust https://rustup.rs/
rustup default 1.75.0
rustup target add aarch64-linux-android
rustup target add armv7-linux-androideabi
rustup target add i686-linux-android
rustup target add x86_64-linux-android
