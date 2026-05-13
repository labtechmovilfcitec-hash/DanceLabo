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
    private string lastData = "";

    public string GetLastData()
    {
        return lastData;
    }

    void Start()
    {
        udpClient = new UdpClient();
        udpClient.Connect(host, port);
        isRunning = true;

        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();

        // Enviar un mensaje inicial para que el servidor Python conozca nuestro endpoint (IP y puerto)
        SendMessageToPython("HELLO_FROM_UNITY");
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
            }
            catch (System.Exception e)
            {
                if(isRunning) Debug.LogError("Error UDP: " + e.ToString());
            }
        }
    }

    public void SendMessageToPython(string message)
    {
        try
        {
            byte[] data = Encoding.UTF8.GetBytes(message);
            udpClient.Send(data, data.Length);
        }
        catch (System.Exception e)
        {
            Debug.LogError("Error enviando mensaje UDP: " + e.ToString());
        }
    }

    void OnApplicationQuit()
    {
        isRunning = false;
        if (receiveThread != null) receiveThread.Abort();
        if (udpClient != null) udpClient.Close();
    }
}
