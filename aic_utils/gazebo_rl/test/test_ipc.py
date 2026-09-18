import threading
import pytest

from gazebo_rl.ipc import IPCServer, JsonLineConnection, connect_with_retry


@pytest.mark.parametrize("transport", ["tcp", "unix"])
def test_ipc_send_receive_round_trip(transport, tmp_path):
    path = tmp_path / "ipc.sock" if transport == "unix" else None
    server = IPCServer(unix_path=path)
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
    client = connect_with_retry("127.0.0.1", server.port, timeout_sec=2.0,
                                unix_path=str(path) if path else None)
    client.send("observation", {"step_count": 1})
    msg = client.recv(timeout_sec=2.0)
    client.close()
    thread.join(timeout=2.0)
    server.close()

    assert received == {"type": "observation", "payload": {"step_count": 1}}
    assert msg.type == "action"
    if path:
        assert not path.exists()


def test_partial_message_survives_timeout_and_preserves_next_message():
    import socket
    reader, writer = socket.socketpair()
    connection = JsonLineConnection(reader)
    try:
        writer.sendall(b'{"type":"observation","payload":')
        with pytest.raises(TimeoutError):
            connection.recv(timeout_sec=0.01)
        writer.sendall(b'{"step":1}}\n{"type":"done","payload":{}}\n')
        assert connection.recv(timeout_sec=1).payload == {"step": 1}
        assert connection.recv(timeout_sec=1).type == "done"
    finally:
        connection.close()
        writer.close()


def test_large_image_message_round_trip():
    import socket
    reader, writer = socket.socketpair()
    sender, receiver = JsonLineConnection(writer), JsonLineConnection(reader)
    payload = {"data_b64": "a" * 500_000}
    thread = threading.Thread(target=lambda: sender.send("observation", payload))
    try:
        thread.start()
        assert receiver.recv(timeout_sec=2).payload == payload
        thread.join(timeout=2)
        assert not thread.is_alive()
    finally:
        sender.close()
        receiver.close()
