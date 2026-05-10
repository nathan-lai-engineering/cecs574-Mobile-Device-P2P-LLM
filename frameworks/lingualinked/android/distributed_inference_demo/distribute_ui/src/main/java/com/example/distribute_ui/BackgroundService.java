package com.example.distribute_ui;
import android.app.Service;
import android.content.Intent;
import android.os.Environment;
import android.os.IBinder;
import android.util.Log;
import androidx.annotation.Nullable;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;
import com.example.SecureConnection.Communication;
import com.example.SecureConnection.Config;
import com.example.SecureConnection.Dataset;
import com.example.SecureConnection.LoadBalance;
import org.greenrobot.eventbus.EventBus;
import org.greenrobot.eventbus.Subscribe;
import org.greenrobot.eventbus.ThreadMode;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.Properties;

public class BackgroundService extends Service {
    public static double[] results;
    public static final String TAG = "Lingual_backend";
    private String role = "worker";
    private boolean need_monitor = false;
    private final boolean running_classification = false;
    private boolean shouldStartInference = false;
    private boolean runningStatus = false;
    private volatile boolean prepareEventReceived = false;
    private boolean messageStatus = false;
    public static boolean isServiceRunning = false;

    private String messageContent = "";

    @Subscribe(threadMode = ThreadMode.BACKGROUND)
    public void onRunningStatus(Events.RunningStatusEvent event){
        runningStatus = event.isRunning;
        prepareEventReceived = true;
        System.out.println("Running Status is: "+runningStatus);
    }

    @Subscribe(threadMode = ThreadMode.BACKGROUND)
    public void onMessageSentEvent(Events.messageSentEvent event) {
        messageStatus = event.messageSent;
        messageContent = event.messageContent;
    }

    @Subscribe(threadMode = ThreadMode.BACKGROUND)
    public void onEnterChatEvent(Events.enterChatEvent event) {
        shouldStartInference = event.enterChat;
    }

    private String getServerIPAddress() {
        String serverIP = "";
        Properties properties = new Properties();
        try {
            InputStream inputStream = getAssets().open("config.properties");
            properties.load(inputStream);
            serverIP = properties.getProperty("server_ip");
            inputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return serverIP;
    }

    private String getDeviceIPOverride() {
        Properties properties = new Properties();
        try {
            InputStream inputStream = getAssets().open("config.properties");
            properties.load(inputStream);
            inputStream.close();
            return properties.getProperty("device_ip", "").trim();
        } catch (IOException e) {
            return "";
        }
    }

    private boolean isModelDirectoryEmpty(String modelPath) {
        File modelDir = new File(modelPath + "/device");
        if (modelDir.isDirectory()) {
            String[] files = modelDir.list();
            return files == null || files.length == 0;
        }
        // Return true if it's not a directory, indicating "empty" in this context.
        return true;
    }

    private void updateIsDirEmpty(boolean isDirEmpty) {
        // Update the repository with the new value
        DataRepository.INSTANCE.setIsDirEmpty(isDirEmpty);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.d(TAG, "background service started");
        int id = 0;
        if (intent != null && intent.hasExtra("role")) {
            id = intent.getIntExtra("role", 0);
        }
        if (id == 1) {
            role = "header";
        }
        Log.d(TAG, "role is " + role);

        String modelName = "";
        if (intent != null && intent.hasExtra("model")) {
            modelName = intent.getStringExtra("model");
            System.out.println("model name is: "+ modelName);
        }

        ExecutorService executor = Executors.newSingleThreadExecutor();
        String finalModelName = modelName;
        executor.submit(() -> {
            // One-time setup: pin the device IP. Config.local is static and survives
            // across session retries, so we only need to set it once.
            String server_ip = getServerIPAddress();
            String deviceIPOverride = getDeviceIPOverride();
            if (!deviceIPOverride.isEmpty()) {
                Config.local = deviceIPOverride;
                Log.d(TAG, "device IP overridden to: " + deviceIPOverride);
            }

            // Outer retry loop — re-registers with the coordinator after each session
            // ends (whether this device was selected or rejected as a standby).
            while (true) {
                // Fresh Communication + Config per session so ZMQ sockets are clean.
                Config cfg = new Config(server_ip, 23456, 7, 0.7f);
                Communication com = new Communication(cfg);
                Communication.loadBalance = new LoadBalance(com, cfg);
                com.param.modelPath = getFilesDir() + "";
                prepareEventReceived = false;
                runningStatus = false;

                // 1. Register with coordinator.
                if (role.equals("header")) {
                    need_monitor = com.sendIPToServer(role, finalModelName);
                } else {
                    need_monitor = com.sendIPToServer(role, "");
                }

                // 2. Start device monitor if coordinator requests hardware metrics.
                if (need_monitor) {
                    Intent broadcastIntent = new Intent();
                    broadcastIntent.setAction("START_MONITOR");
                    LocalBroadcastManager.getInstance(this).sendBroadcast(broadcastIntent);
                    sendBroadcast(broadcastIntent);
                    Log.d(TAG, "broadcast sent by backgroundService");
                }

                // 3. Download model shard and complete lifecycle (Ready→Prepare→Start).
                //    prepareEventReceived is set by onRunningStatus on either success or rejection.
                com.runPrepareThread();
                while (!prepareEventReceived) {
                    try {
                        Thread.sleep(1000);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return null;
                    }
                }

                if (!runningStatus) {
                    // Coordinator assigned this session's shards to other devices.
                    // Stand by and retry when the next session opens.
                    Log.d(TAG, "No shard assigned this session — standing by. Retrying in 30s...");
                    try {
                        Thread.sleep(30_000);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return null;
                    }
                    continue;
                }

                boolean isDirEmpty = isModelDirectoryEmpty(com.param.modelPath);
                if (!isDirEmpty) {
                    System.out.println("Prepare is Finished.");
                    if (cfg.isHeader()) {
                        updateIsDirEmpty(isDirEmpty);
                    }
                    System.out.println("Should start the inference: " + shouldStartInference);
                }

                // 4. Header waits for the user to navigate to the chat screen.
                if (cfg.isHeader()) {
                    while (!shouldStartInference) {
                        try {
                            Thread.sleep(1000);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            return null;
                        }
                    }
                }

                // 5. Run inference for this session.
                if (shouldStartInference && cfg.isHeader()) {
                    com.param.classes = new String[]{"Negative", "Positive"};
                    Dataset dataset = null;

                    while (com.param.numSample <= 0)
                        Thread.sleep(1000);

                    ArrayList<String> test_input = new ArrayList<>();

                    while (!messageStatus) {
                        try {
                            Thread.sleep(1000);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            return null;
                        }
                    }

                    new Thread(() -> {
                        int j = 0;
                        String userinput = "";
                        while (j < com.param.numSample) {
                            if (messageContent.equals(userinput)) {
                                try {
                                    Thread.sleep(1000);
                                } catch (InterruptedException e) {
                                    throw new RuntimeException(e);
                                }
                            } else {
                                System.out.println("New user input");
                                System.out.println("***************" + messageContent);
                                userinput = messageContent;
                                test_input.add(userinput);
                                j++;
                            }
                        }
                    }).start();

                    int corePoolSize = 2;
                    int maximumPoolSize = 2;
                    int keepAliveTime = 500;
                    double startTime = System.nanoTime();
                    try {
                        Log.d(TAG, "communication starts to running");
                        com.running(corePoolSize, maximumPoolSize, keepAliveTime, test_input);
                    } catch (IOException | InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                    results = com.timeUsage;

                    if (running_classification) {
                        if (cfg.isHeader()) {
                            double accuracy = 0.0;
                            for (int i = 0; i < com.logits.size(); i++) {
                                int pred = binaryClassify(com.logits.get(i));
                                int truth = dataset.labels.get(i).equals("positive") ? 1 : 0;
                                if (pred == truth) {
                                    accuracy += 1;
                                }
                            }
                            Log.d(TAG, "Task Accuracy: " + (accuracy / com.logits.size()));
                        }
                    }
                    Log.d(TAG, "Results Computation Time: " + (System.nanoTime() - startTime) / 1000000000.0);

                } else if (!cfg.isHeader()) {
                    com.param.classes = new String[]{"Negative", "Positive"};
                    while (com.param.numSample <= 0)
                        Thread.sleep(1000);
                    ArrayList<String> test_input = new ArrayList<>();
                    int corePoolSize = 2;
                    int maximumPoolSize = 2;
                    int keepAliveTime = 500;
                    try {
                        Log.d(TAG, "communication starts to running");
                        com.running(corePoolSize, maximumPoolSize, keepAliveTime, test_input);
                    } catch (IOException | InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                    results = com.timeUsage;
                }

                // 6. Session complete — reset per-session state, then loop back to
                //    re-register for the next session.
                Log.d(TAG, "Session complete. Reconnecting in 10s...");
                shouldStartInference = false;
                messageStatus = false;
                messageContent = "";
                try {
                    Thread.sleep(10_000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return null;
                }
            }
        });

        return START_STICKY; // This tells the system to restart the service if it gets killed due to resource constraints.
    }

    private void loadZipFile() {
        File sourceFile = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "device.zip");
        File destinationFile = new File(getFilesDir() + "/device.zip");

        if (!destinationFile.getParentFile().exists()) {
            destinationFile.getParentFile().mkdirs(); // Create the parent path if it doesn't exist
        }

        Log.d(TAG, "SourceFile: " + sourceFile.getAbsolutePath());
        Log.d(TAG, "DestFile: " + destinationFile.getAbsolutePath());

        try (InputStream in = new FileInputStream(sourceFile)) {
            try (OutputStream out = new FileOutputStream(destinationFile)) {
                // Transfer bytes from in to out
                byte[] buf = new byte[1024];
                int len;
                while ((len = in.read(buf)) > 0) {
                    out.write(buf, 0, len);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
            // Handle the exception
        }

    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
    @Override
    public void onCreate() {
        super.onCreate();
        isServiceRunning = true;
        EventBus.getDefault().register(this);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        isServiceRunning = false;
        EventBus.getDefault().unregister(this);
    }

    public native int binaryClassify(byte[] data);

}
