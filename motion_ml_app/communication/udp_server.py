import socket
import json
import threading

class UDPServer:
    def __init__(self, host='127.0.0.1', port=5005):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client_address = None
        self.running = False

    def start(self):
        self.sock.bind((self.host, self.port))
        self.running = True
        print(f"Servidor UDP escuchando en {self.host}:{self.port}")
        
        # Hilo para recibir mensajes de Unity sin bloquear la UI
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()

    def _receive_loop(self):
        while self.running:
            try:
                data, address = self.sock.recvfrom(1024)
                self.client_address = address  # Guardamos la ip/puerto del cliente
                message = data.decode('utf-8')
                print(f"Mensaje recibido de Unity: {message}")
                # TODO: Procesar el comando (ej: "Aprende", "Ejecuta") y notificar al modulo ML
            except OSError:
                break

    def send_pose(self, pose_data):
        if self.client_address and self.running:
            try:
                message = json.dumps(pose_data).encode('utf-8')
                self.sock.sendto(message, self.client_address)
            except Exception as e:
                print(f"Error enviando pose: {e}")

    def stop(self):
        self.running = False
        self.sock.close()
