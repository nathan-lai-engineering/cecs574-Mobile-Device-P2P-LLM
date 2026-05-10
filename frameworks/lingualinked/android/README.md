# Android — LinguaLinked Header

The Android app (`distribute_ui`) runs on the Android Studio emulator as the **header node** in distributed inference. It handles:
- Registering with the coordinator
- Receiving the model shard assigned to the header
- Accepting user chat input and driving token generation
- Streaming decoded tokens back to the chat UI

## Project location

Open this directory in Android Studio:

```
<repo-root>/frameworks/lingualinked/android/distributed_inference_demo
```

On this machine:
```
c:\CSULB\cecs574-Mobile-Device-P2P-LLM\frameworks\lingualinked\android\distributed_inference_demo
```

## Configuration file

Before building, set the coordinator address and device IP in:

```
c:\CSULB\cecs574-Mobile-Device-P2P-LLM\frameworks\lingualinked\android\distributed_inference_demo\distribute_ui\src\main\assets\config.properties
```

See `frameworks/lingualinked/README.md` for what values to set.

## Build and run

1. Open the project in Android Studio (`File → Open` → select the `distributed_inference_demo` directory)
2. Wait for Gradle sync to complete
3. Create an AVD with at least 6 GB RAM (Device Manager → Create Virtual Device)
4. Select the AVD in the device dropdown and click the green Run button
5. In the app: select **Header**, select **TinyLlama**, tap **Next**

See `frameworks/lingualinked/README.md` for the full setup sequence including ADB port forwarding.

## Native libraries

The `distribute_ui/src/main/cpp/` directory contains the C++ JNI layer:
- ONNX Runtime inference calls
- HuggingFace tokenizer (via `tokenizers-cpp`)
- ZMQ tensor exchange between devices

These are compiled automatically by Android Studio as part of the CMake build. No manual compilation step is needed.
