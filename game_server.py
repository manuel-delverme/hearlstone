import envs.simple_HSenv
import socket


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 31337))
    s.listen(1)

    environment = envs.simple_HSenv.simple_HSEnv()
    state = environment.reset()
    conn, addr = s.accept()

    conn.sendall(state)
    with conn:
        print('Connected by', addr)
        while True:
            action = conn.recv(1)
            if not action:
                raise Exception("NO DATA!")
            environment.step(action)
            conn.sendall(state)


