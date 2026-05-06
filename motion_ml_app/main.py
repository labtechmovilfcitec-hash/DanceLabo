import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow
from communication.udp_server import UDPServer

if __name__ == "__main__":
    # Iniciar servidor UDP en hilo separado
    udp_server = UDPServer()
    udp_server.start()

    # Iniciar UI
    app = QApplication(sys.argv)
    window = MainWindow(udp_server=udp_server)
    window.show()
    sys.exit(app.exec())
