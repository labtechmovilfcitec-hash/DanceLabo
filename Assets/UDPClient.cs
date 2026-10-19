using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public class UDPClient : MonoBehaviour
{
    public string host = "127.0.0.1";
    public int port = 5005;

    private UdpClient udpClient;
    private Thread receiveThread;
    private bool isRunning;
    private volatile string lastData = "";
    private volatile int dataVersion = 0;

    public string GetLastData()
    {
        return lastData;
    }

    /// <summary>
    /// Versión de los datos recibidos. Incrementa con cada paquete UDP.
    /// Permite a MixamoAnimator detectar nuevos datos con una comparación numérica ultrarrápida.
    /// </summary>
    public int DataVersion => dataVersion;

    void Start()
    {
        udpClient = new UdpClient();
        try
        {
            udpClient.Client.ReceiveBufferSize = 65536;
        }
        catch {}

        // En Windows: Desactivar WSAECONNRESET (10054) para que el socket UDP
        // no se rompa si Python aún no está iniciado o si se reinicia.
        try
        {
            const int SIO_UDP_CONNRESET = -1744830452;
            udpClient.Client.IOControl((IOControlCode)SIO_UDP_CONNRESET, new byte[] { 0, 0, 0, 0 }, null);
        }
        catch {}

        udpClient.Connect(host, port);
        isRunning = true;

        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();

        // Enviar handshake periódicamente (cada 1s) para que Python siempre conozca
        // nuestro puerto efímero, sin importar qué aplicación se abra primero ni si Python se reinicia.
        InvokeRepeating(nameof(SendHeartbeat), 0.05f, 1.0f);
    }

    private void SendHeartbeat()
    {
        if (isRunning)
        {
            SendMessageToPython("HELLO_FROM_UNITY");
        }
    }

    private void ReceiveData()
    {
        while (isRunning)
        {
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                byte[] data = udpClient.Receive(ref anyIP);
                string text = Encoding.UTF8.GetString(data);
                
                // NOTA: Los datos llegan en otro hilo. Para actualizar GameObjects 
                // Guardamos los datos para aplicarlos en el Update() (Main Thread).
                lastData = text;
                unchecked { dataVersion++; }
            }
            catch (System.Exception e)
            {
                if (isRunning)
                {
                    // Si ocurre un error temporal, pausar brevemente para no saturar CPU
                    Thread.Sleep(5);
                }
            }
        }
    }

    public void SendMessageToPython(string message)
    {
        try
        {
            if (udpClient != null && udpClient.Client != null)
            {
                byte[] data = Encoding.UTF8.GetBytes(message);
                udpClient.Send(data, data.Length);
            }
        }
        catch (System.Exception)
        {
            // Silencioso si Python aún no está abierto
        }
    }

    void OnDisable()
    {
        CancelInvoke();
    }

    void OnDestroy()
    {
        CancelInvoke();
        isRunning = false;
        try { if (receiveThread != null && receiveThread.IsAlive) receiveThread.Abort(); } catch {}
        try { if (udpClient != null) udpClient.Close(); } catch {}
    }

    void OnApplicationQuit()
    {
        CancelInvoke();
        isRunning = false;
        try { if (receiveThread != null && receiveThread.IsAlive) receiveThread.Abort(); } catch {}
        try { if (udpClient != null) udpClient.Close(); } catch {}
    }
}
