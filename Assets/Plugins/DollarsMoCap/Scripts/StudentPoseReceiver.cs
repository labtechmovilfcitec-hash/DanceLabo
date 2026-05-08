using System;
using System.Collections.Generic;
using UnityEngine;
using uOSC;

namespace Dollars
{
    [RequireComponent(typeof(uOscServer))]
    public class StudentPoseReceiver : MonoBehaviour
    {
        [Header("Configuraci�n de Red")]
        public int port = 39540;

        [Header("Estado del Estudiante (Lectura)")]
        public bool IsStudentDetected = false;
        public float DetectionConfidence = 0.0f;

        // Almacenamiento de pose
        public Dictionary<string, Vector3> studentBonePositions = new Dictionary<string, Vector3>();
        public Dictionary<string, Quaternion> studentBoneRotations = new Dictionary<string, Quaternion>();

        private uOscServer server;
        private float lastPacketTime;
        private const float TimeoutThreshold = 1.0f; // Tiempo para considerar que se perdi� la detecci�n

        void Awake()
        {
            server = GetComponent<uOscServer>();
            server.port = port;
        }

        void OnEnable()
        {
            server.onDataReceived.AddListener(OnDataReceived);
        }

        void OnDisable()
        {
            server.onDataReceived.RemoveListener(OnDataReceived);
        }

        void Update()
        {
            // Si no recibimos paquetes en 1 segundo, marcamos como no detectado
            if (Time.time - lastPacketTime > TimeoutThreshold)
            {
                IsStudentDetected = false;
                DetectionConfidence = 0f;

                //Limpiar datos viejos
                studentBonePositions.Clear();
                studentBoneRotations.Clear();
            }
        }

        private void OnDataReceived(Message message)
        {
            if (message.address == null || message.values == null) return;

            // Actualizar tiempo de �ltimo paquete recibido
            lastPacketTime = Time.time;

            // 1. Recibir Pose del Estudiante
            if (message.address == "/Student/Bone/Pos")
            {
                ProcessBonePose(message);
            }
            // 2. Recibir Visibilidad/Confianza
            else if (message.address == "/Student/Visibility")
            {
                ProcessVisibility(message);
            }
        }

        private void ProcessBonePose(Message message)
        {
            // Estructura esperada: [boneName, px, py, pz, rx, ry, rz, rw]
            if (message.values.Length >= 8)
            {
                try
                {
                    string boneName = message.values[0].ToString();
                    Vector3 pos = new Vector3((float)message.values[1], (float)message.values[2], (float)message.values[3]);
                    Quaternion rot = new Quaternion((float)message.values[4], (float)message.values[5], (float)message.values[6], (float)message.values[7]);

                    studentBonePositions[boneName] = pos;
                    studentBoneRotations[boneName] = rot;

                    IsStudentDetected = true; // Si hay huesos, hay detecci�n
                }
                catch (Exception)
                {
                    // Ignorar errores de formato para evitar romper la ejecuci�n
                }
            }
        }

        private void ProcessVisibility(Message message)
        {
            // Estructura esperada: [float confidence]
            if (message.values.Length >= 1 && message.values[0] is float)
            {
                DetectionConfidence = (float)message.values[0];
                IsStudentDetected = DetectionConfidence > 0.1f; // Umbral m�nimo
            }
        }
    }
}