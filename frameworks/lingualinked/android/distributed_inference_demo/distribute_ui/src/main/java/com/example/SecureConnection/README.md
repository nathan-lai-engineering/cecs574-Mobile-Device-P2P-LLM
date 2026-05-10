# SecureConnection — Android Package

Java/Kotlin communication layer for the Android header node. Mirrors the protocol in the Python `SecureConnection/` package on the coordinator side.

## Key classes

| Class | Role |
|---|---|
| `Communication.java` | Entry point — manages ZMQ context, sockets, and the `running()` inference loop |
| `Client.java` | Handles the full device lifecycle: Ready → Open → Prepare → Initialized → Start → Running → Finish → Close. Receives the model shard from the coordinator and posts `RunningStatusEvent` when the lifecycle completes (success or rejection). |
| `Config.java` | Holds coordinator address, ports, device role (header/worker/tailer), and the IP graph from the coordinator. `Config.local` is the device's own registered IP. |
| `LoadBalance.java` | Manages ZMQ DEALER sockets to neighboring devices for tensor passing during inference |
| `Dataset.java` | Input data helpers |

## How it fits into the app

`BackgroundService.java` (in `com.example.distribute_ui`) starts a background thread that:
1. Reads `server_ip` and `device_ip` from `assets/config.properties`
2. Calls `Communication.sendIPToServer()` to register with the coordinator
3. Calls `Communication.runPrepareThread()` which drives the `Client.java` lifecycle
4. Waits for `RunningStatusEvent` — `true` means this device received a shard and is ready; `false` means it was rejected (no shard assigned) and will retry in 30 seconds
5. After receiving a shard, waits for the user to navigate to the chat screen, then calls `Communication.running()` to start inference

## Configuration

The coordinator address and device IP come from:
```
distribute_ui/src/main/assets/config.properties
```

Full path on this machine:
```
c:\CSULB\cecs574-Mobile-Device-P2P-LLM\frameworks\lingualinked\android\distributed_inference_demo\distribute_ui\src\main\assets\config.properties
```

This file is read at runtime by `BackgroundService.java` using `getAssets().open("config.properties")`. Changing it requires rebuilding the APK.
