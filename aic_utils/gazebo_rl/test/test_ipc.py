import threading

from gazebo_rl.ipc import IPCServer, connect_with_retry


def test_ipc_send_receive_round_trip():
    server = IPCServer()
    received = {}

    def server_thread():
        conn = server.accept(timeout_sec=2.0)
        msg = conn.recv(timeout_sec=2.0)
        received["type"] = msg.type
        received["payload"] = msg.payload
        conn.send("action", {"action": [0, 0, 0, 0, 0, 0]})
        conn.close()

    thread = threading.Thread(target=server_thread)
    thread.start()
    client = connect_with_retry("127.0.0.1", server.port, timeout_sec=2.0)
    client.send("observation", {"step_count": 1})
    msg = client.recv(timeout_sec=2.0)
    client.close()
    thread.join(timeout=2.0)
    server.close()

    assert received == {"type": "observation", "payload": {"step_count": 1}}
    assert msg.type == "action"
