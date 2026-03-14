# llama instructions

## compiling (tested with g++)
cmake -B build
cmake --build build -j --target llama-server llama-cli

## execution
./llama-cli -m <path/to/your/model.gguf> -p "Your prompt here"
