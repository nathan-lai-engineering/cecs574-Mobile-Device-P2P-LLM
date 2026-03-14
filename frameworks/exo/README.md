
# llama instructions

sudo apt update

# Install Node.js and npm
sudo apt install nodejs npm

# Install uv (note: restarting the shell might be required)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Rust (using rustup)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly

# Build dashboard
cd exo/dashboard
npm i
npm run build
cd ..

# Install python

uv python install 3.14 

# Install G++
sudo apt install g++-12
sudo apt install build-essential

# Install Nvidia CUDA libary (if needed)
wget https://developer.download.nvidia.com/compute/cuda/13.2.0/local_installers/cuda-repo-debian12-13-2-local_13.2.0-595.45.04-1_amd64.deb
sudo dpkg -i cuda-repo-debian12-13-2-local_13.2.0-595.45.04-1_amd64.deb
sudo cp /var/cuda-repo-debian12-13-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-2


# Run exo
uv run exo