import socket
import json
import time
from threading import Thread

HOST = '172.16.0.220'  # Server IP
PORT = 7450        # Server port
LPORT = 7451

def listen_thread():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, LPORT))
        srv.listen(1)
        print(f"Python Echo Listening on {HOST}:{PORT}")
        try:
            while True:
                conn, addr = srv.accept()
                try:
                    with conn, conn.makefile("r", encoding="utf-8", newline="\n") as f:
                        for line in f:
                            print(f"Received: {line}")
                except Exception as e:
                    print(f"Error: {e}")
        except KeyboardInterrupt:
            print("Exitting Server")
            return

def send_repeat():
    # Create a TCP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print(f"Connected to {HOST}:{PORT}")

        counter = 0
        while True:
            # Construct a sample JSON message
            message = {
                "cmd": "move",
                "unit": counter,
                "x": 100 + counter * 10,
                "z": 200 + counter * 5
            }

            # Convert to JSON string and append newline
            json_str = json.dumps(message) + "\n"

            # Send it over the socket
            s.sendall(json_str.encode('utf-8'))
            print("Sent:", json_str.strip())

            counter += 1
            time.sleep(1)  # Wait 1 second

def main():
    st = Thread(target=send_repeat, daemon=True)
    # st.start()
    listen_thread()

if __name__ == "__main__":
    main()
