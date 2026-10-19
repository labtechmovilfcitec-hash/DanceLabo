import socket
import json
import threading

class UDPServer:
    def __init__(self, host='127.0.0.1', port=5005):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # En Windows: desactivar WSAECONNRESET (10054)
        if hasattr(socket, "SIO_UDP_CONNRESET"):
            try:
                self.sock.ioctl(socket.SIO_UDP_CONNRESET, False)
            except Exception:
                pass

        self.client_address = None
        self.running = False
        self._sent_count = 0
        self._warned_waiting = False

    def start(self):
        try:
            self.sock.bind((self.host, self.port))
            print(f"[UDPServer] Servidor UDP escuchando en {self.host}:{self.port}")
        except OSError as e:
            print(f"[UDPServer] ERROR: No se pudo vincular al puerto {self.port} ({e}).")
            print("[UDPServer] Asegúrate de que no haya otra instancia de DanceLabo o Python abierta.")
            return

        self.running = True
        
        # Hilo para recibir mensajes de Unity sin bloquear la UI
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()

    def _receive_loop(self):
        while self.running:
            try:
                data, address = self.sock.recvfrom(2048)

                # Descartar paquetes que provengan de nuestro propio puerto de escucha
                if address[1] == self.port:
                    continue

                # Actualizar la dirección del cliente Unity
                if self.client_address != address:
                    self.client_address = address
                    print(f"[UDPServer] Unity conectado y listo desde {address}")

                message = data.decode('utf-8', errors='ignore')
                if message != "HELLO_FROM_UNITY":
                    print(f"[UDPServer] Mensaje de Unity ({address}): {message}")
            except ConnectionResetError:
                continue
            except OSError:
                if not self.running:
                    break
                continue

    def send_pose(self, pose_data):
        if not self.running or not pose_data:
            return
        if self.client_address is None:
            if not self._warned_waiting:
                print("[UDPServer] Esperando señal de Unity (presiona Play en Unity)...")
                self._warned_waiting = True
            return

        self._warned_waiting = False
        try:
            message = json.dumps(pose_data).encode('utf-8')
            self.sock.sendto(message, self.client_address)
            self._sent_count += 1
            if self._sent_count % 120 == 1:
                print(f"[UDP] Enviando poses activamente a Unity {self.client_address} (frame #{self._sent_count})")
        except Exception as e:
            if self._sent_count % 120 == 1:
                print(f"[UDP] Error enviando pose: {e}")

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass
