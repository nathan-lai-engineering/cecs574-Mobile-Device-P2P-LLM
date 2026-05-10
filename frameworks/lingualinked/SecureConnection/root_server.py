''''''
import json
import time
from threading import Thread

"""
    R: Ready
        client -> root; Show the status of edge device
    O: Open
        root -> client; The information of this task, e.g: Training/Inference/Task name, etc. 
    P: Prepare
        root -> client; Send the decentralized model and training/Inference code to clients. 
    I:  Initialized
        client -> root; Models are initialized and training/Inference is ready
    S: Start
        root -> client; Start training/Inference and Data transmission)    
    F: Finish
        client -> root; Finish training/Inference
    C: Close
        root -> client; Close the connection
"""

def communication_open_close(sender, config, status, conditions, lock, open=True, abort_event=None):
    """Handle one device through the full Ready→Open→Prepare→Initialized→Start→Running→Finish→Close
    lifecycle.  abort_event (threading.Event) can be set externally to unblock a stuck session."""

    COND_POLL_S = 10  # how often to check abort inside condition waits

    def recv_next():
        """Return the next multipart message, or None if abort fires."""
        while abort_event is None or not abort_event.is_set():
            with lock[0]:
                if sender.poll(1000):
                    return sender.recv_multipart()
        return None

    ## Status: Ready, Open, Prepare, Initialized, Start, Running, Finish
    while True:
        print('enter communication open close')
        info = recv_next()
        if info is None:
            print('[Server] abort_event set — exiting lifecycle thread.')
            return
        client_id = info[0]
        msg = info[1]
        print(client_id + msg)
        ## Ready
        if open and msg == b'Ready':
            ## Open
            if len(info) != 3:
                print("Error")
                continue

            node_ip = info[2]
            if node_ip not in config["file_path"]:
                # Device registered but has no shard assigned — reject and exit this thread.
                # Status is NOT updated so the barriers for valid devices are unaffected.
                print(f"[Server] Device {node_ip} has no shard — sending Close.")
                with lock[1]:
                    sender.send_multipart([client_id, b'Close'])
                return

            config["ids"][client_id] = node_ip
            print(config["ids"])

            status[client_id] = b'Ready'

            # Send each field as a separate message so Android's per-recv() reads succeed.
            # JeroMQ DEALER treats each send_multipart as one message; packing everything
            # into a single 9-frame multipart causes recv() to block after the first frame.
            # lock[1] prevents concurrent sends from racing on the shared ROUTER socket.
            with lock[1]:
                sender.send_multipart([client_id, b'Open'])
                sender.send_multipart([client_id, config["graph"]])
                sender.send_multipart([client_id, config["session_index"]])
                sender.send_multipart([client_id, config["task_type"]])
                sender.send_multipart([client_id, config["core_pool_size"]])
                sender.send_multipart([client_id, config["num_sample"]])
                sender.send_multipart([client_id, config["max_length"]])
                sender.send_multipart([client_id, json.dumps(config["dependency"]).encode()])

            status[client_id] = b'Open'
            print(f"Status: Open {config['ids'][client_id]}")

            with conditions[0]:
                while not check_status(status, config, b"Open"):
                    conditions[0].wait(timeout=COND_POLL_S)
                    if abort_event is not None and abort_event.is_set():
                        return
                conditions[0].notify_all()

            ## Prepare
            with lock[1]:
                communication_prepare(sender, config, client_id, status)

            print(f"Status: Prepare {config['ids'][client_id]}")

        ## Initialized
        elif msg == b'Initialized':
            status[client_id] = b'Initialized'
            print(f"Status: Initialized {config['ids'][client_id]}")

            with conditions[1]:
                while not check_status(status, config, b"Initialized"):
                    conditions[1].wait(timeout=COND_POLL_S)
                    if abort_event is not None and abort_event.is_set():
                        return
                conditions[1].notify_all()

            ## Start — acquire send lock OUTSIDE the condition block to avoid
            ## holding the condition's internal lock while writing to the socket.
            with lock[1]:
                sender.send_multipart([client_id, b"Start"])
            status[client_id] = b'Start'

            print(f"Status: Start {config['ids'][client_id]}")

        elif msg == b"Running":
            pass
        elif msg == b'Finish':
            status[client_id] = b'Close'
            with conditions[2]:
                while not check_status(status, config, b"Close"):
                    conditions[2].wait(timeout=COND_POLL_S)
                    if abort_event is not None and abort_event.is_set():
                        return
                conditions[2].notify_all()

            with lock[1]:
                sender.send_multipart([client_id, b"Close"])
            print(f"Close {config['ids'][client_id]}")
            break

def send_model_file(path, sock, client_id, chunked=True, chunk_size=10*1024*1024):
    if not chunked:
        with open(path, 'rb') as f:
            data = f.read()
            sock.send_multipart([client_id, data])
            print("Data is sent")
    else:
        with open(path, 'rb') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    sock.send_multipart([client_id, b''])
                    break
                sock.send_multipart([client_id, chunk])
            print("Data is sent")

def communication_prepare(sender, config, client_id, status):
    sender.send_multipart([client_id, b'Prepare'])
    node_ip = config["ids"][client_id]
    sender.send_multipart([client_id, str(config["skip_model_transmission"]).encode()])

    if not config["skip_model_transmission"]:   ## Assume data is received on the machine
        print(f"send {config['file_path'][node_ip]} to {node_ip}")

        # onnx sends multiple files, which should be a zip
        if config["onnx"]:
            send_model_file(config["file_path"][node_ip], sender, client_id)
        else:
            send_model_file(config["file_path"][node_ip], sender, client_id)

        # # transmit tokenizer to header
        # if config["head_node"] == node_ip:
        #     print(f"send {config['file_path'][b'tokenizer']} to {node_ip}")
        #     send_model_file(config["file_path"][b"tokenizer"], sender, client_id)

    status[client_id] = b"Prepare"

def communication_data_transmission(sender, num_devices, head_client_id, status):
    while check_status(status, num_devices, b"Start"):
        pass


def communication_result_transmission(sender, result, num_devices, tail_client_id, status):
    # while check_status(status, num_devices, b"Finish"):
    sender.send_multipart([b"res", result])
    pass


all_status = {b"Ready":  0,
              b"Open":   1,
              b"Prepare":2,
              b"Initialized": 3,
              b"Start":  4,
              b"Running":5,
              b"Finish": 6,
              b"Close": 7}

def check_status(status, config, mode):
    if len(status) != config["num_device"]:
        return False
    for v in status.values():
        if all_status[v] < all_status[mode]:
            return False
    return True


def ConfigCreator(Config, client_id):
    ## Based on the monitor situation
    return Config["graph"]