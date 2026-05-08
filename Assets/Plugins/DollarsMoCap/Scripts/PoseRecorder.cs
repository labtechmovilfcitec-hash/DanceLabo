using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Dollars
{
    public class PoseRecorder : MonoBehaviour
    {
        [Header("Configuraci�n de Grabaci�n")]
        public bool IsRecording = false;
        public string FileName = "RobotRecording";
        public float TargetFPS = 30f;

        [Header("Referencia del Modelo")]
        public Animator TargetAnimator;

        [Header("Estado (Lectura)")]
        public int RecordedFramesCount = 0;

        // Estructuras para JSON
        [Serializable]
        public class BoneData
        {
            public string boneName;
            public Vector3 localPosition;
            public Quaternion localRotation;
        }

        [Serializable]
        public class FrameData
        {
            public float timestamp;
            public List<BoneData> bones = new List<BoneData>();
        }

        [Serializable]
        public class RecordingData
        {
            public List<FrameData> frames = new List<FrameData>();
        }

        private RecordingData currentRecording = new RecordingData();
        private float timer = 0f;
        private bool wasRecording = false;

        void Update()
        {
            // Detectar cuando se detiene la grabaci�n para guardar el archivo
            if (wasRecording && !IsRecording)
            {
                SaveToFile();
            }

            if (IsRecording && TargetAnimator != null)
            {
                timer += Time.deltaTime;

                // Control de 30 FPS
                if (timer >= (1f / TargetFPS))
                {
                    RecordFrame();
                    timer = 0f;
                }
            }

            wasRecording = IsRecording;
        }

        private void RecordFrame()
        {
            FrameData frame = new FrameData
            {
                timestamp = Time.time,
                bones = new List<BoneData>()
            };

            // Capturar todos los huesos Humanoid
            foreach (HumanBodyBones boneType in Enum.GetValues(typeof(HumanBodyBones)))
            {
                if (boneType == HumanBodyBones.LastBone) continue;

                Transform boneTransform = TargetAnimator.GetBoneTransform(boneType);
                if (boneTransform != null)
                {
                    frame.bones.Add(new BoneData
                    {
                        boneName = boneType.ToString(),
                        localPosition = boneTransform.localPosition,
                        localRotation = boneTransform.localRotation
                    });
                }
            }

            currentRecording.frames.Add(frame);
            RecordedFramesCount = currentRecording.frames.Count;
        }

        private void SaveToFile()
        {
            // Cambia esto por la ruta absoluta de tu carpeta de secuencias en el proyecto
            // Ejemplo: "C:/TuProyecto/motion_ml_app/data/sequences"
            string folderPath = Path.Combine(Application.dataPath, "../motion_ml_app/data/sequences");
            
            // Crear la carpeta si no existe
            if (!Directory.Exists(folderPath)) Directory.CreateDirectory(folderPath);

            string path = Path.Combine(folderPath, FileName + "_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".json");

            string json = JsonUtility.ToJson(currentRecording, false);
            File.WriteAllText(path, json);

            Debug.Log($"<color=green>Grabación guardada en carpeta de secuencias: {path}</color>");
            
            currentRecording = new RecordingData();
            RecordedFramesCount = 0;
        }
    }
}