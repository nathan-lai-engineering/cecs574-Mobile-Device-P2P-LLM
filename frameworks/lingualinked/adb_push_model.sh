#!/usr/bin/env bash
# Push TinyLlama ONNX model files to emulator via ADB.
# Run AFTER: emulator resized to 16GB, cold booted, APK installed.

set -e

TMPDIR="C:/Users/NATHAN~1/AppData/Local/Temp/tmpr7m9d8c5"
UUID0="438668fc-1ba4-11f1-a2a0-ba29f5ac257c"
UUID1="5813d25a-1ba4-11f1-a2a0-ba29f5ac257c"
APP="com.example.distribute_ui"
APPFILES="files/device"

echo "=== Creating app device directory ==="
adb shell "run-as $APP mkdir -p $APPFILES"

echo "=== Pushing UUID0 (~300MB) to /sdcard ==="
MSYS_NO_PATHCONV=1 adb push "$TMPDIR/$UUID0" "/sdcard/$UUID0"

echo "=== Pushing UUID1 (~4.1GB) to /sdcard ==="
MSYS_NO_PATHCONV=1 adb push "$TMPDIR/$UUID1" "/sdcard/$UUID1"

echo "=== Copying UUID0 from /sdcard to app private dir ==="
adb shell "cat /sdcard/$UUID0 | run-as $APP sh -c 'cat > $APPFILES/$UUID0 && echo UUID0_done'"

echo "=== Copying UUID1 from /sdcard to app private dir (may take a while) ==="
adb shell "cat /sdcard/$UUID1 | run-as $APP sh -c 'cat > $APPFILES/$UUID1 && echo UUID1_done'"

echo "=== Pushing small files via base64 ==="
base64 -w 0 "$TMPDIR/module_0.onnx" | adb shell "run-as $APP sh -c 'base64 -d > $APPFILES/module_0.onnx && echo module_0_done'"
base64 -w 0 "$TMPDIR/module_1.onnx" | adb shell "run-as $APP sh -c 'base64 -d > $APPFILES/module_1.onnx && echo module_1_done'"
base64 -w 0 "$TMPDIR/tokenizer.json" | adb shell "run-as $APP sh -c 'base64 -d > $APPFILES/tokenizer.json && echo tokenizer_done'"

echo "=== Cleaning up /sdcard ==="
adb shell "rm /sdcard/$UUID0 /sdcard/$UUID1"

echo "=== Verifying files on device ==="
adb shell "run-as $APP ls -lh $APPFILES/"

echo ""
echo "Done! Set MODEL_EXIST_ON_DEVICE = True in root.py and restart."
