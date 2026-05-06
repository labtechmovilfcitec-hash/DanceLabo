using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// REQUIRES: com.unity.nuget.newtonsoft-json package in Unity (via Package Manager)
using Newtonsoft.Json.Linq; 

public class MixamoAnimator : MonoBehaviour
{
    public UDPClient udpClient;
    public bool mirrorMode = true; // Invierte el eje X visualmente para comportarse como espejo
    
    [Header("Mixamo Bones")]
    public Transform leftUpLeg;
    public Transform leftLeg;
    public Transform leftFoot;
    public Transform rightUpLeg;
    public Transform rightLeg;
    public Transform rightFoot;
    
    public Transform leftArm;
    public Transform leftForeArm;
    public Transform leftHand;
    public Transform rightArm;
    public Transform rightForeArm;
    public Transform rightHand;
    
    private string lastProcessedData = "";

    private Dictionary<Transform, Quaternion> initialRotations = new Dictionary<Transform, Quaternion>();
    private Dictionary<Transform, Vector3> initialDirections = new Dictionary<Transform, Vector3>();

    void Start()
    {
        // Guardar la T-Pose inicial y la direccion hacia el hueso hijo
        SaveInitialState(leftArm, leftForeArm);
        SaveInitialState(rightArm, rightForeArm);
        SaveInitialState(leftForeArm, leftHand);
        SaveInitialState(rightForeArm, rightHand);
        
        SaveInitialState(leftUpLeg, leftLeg);
        SaveInitialState(rightUpLeg, rightLeg);
        SaveInitialState(leftLeg, leftFoot);
        SaveInitialState(rightLeg, rightFoot);
    }

    void SaveInitialState(Transform bone, Transform child)
    {
        if (bone != null)
        {
            initialRotations[bone] = bone.rotation;
            if (child != null)
            {
                initialDirections[bone] = (child.position - bone.position).normalized;
            }
        }
    }

    void Update()
    {
        if (udpClient == null) return;

        string currentData = udpClient.GetLastData(); 
        
        // Si hay data nueva, la procesamos
        if (currentData != lastProcessedData && !string.IsNullOrEmpty(currentData))
        {
            lastProcessedData = currentData;
            ApplyPoses(currentData);
        }
    }

    void ApplyPoses(string jsonString)
    {
        try
        {
            // Parseamos el JSON que ahora contiene vectores direccionales normalizados
            JObject vectors = JObject.Parse(jsonString);
            
            // Brazos
            ApplyBoneTransform(leftArm, vectors["mixamorig:LeftArm"]);
            ApplyBoneTransform(rightArm, vectors["mixamorig:RightArm"]);
            ApplyBoneTransform(leftForeArm, vectors["mixamorig:LeftForeArm"]);
            ApplyBoneTransform(rightForeArm, vectors["mixamorig:RightForeArm"]);
            
            // Piernas
            ApplyBoneTransform(leftUpLeg, vectors["mixamorig:LeftUpLeg"]);
            ApplyBoneTransform(rightUpLeg, vectors["mixamorig:RightUpLeg"]);
            ApplyBoneTransform(leftLeg, vectors["mixamorig:LeftLeg"]);
            ApplyBoneTransform(rightLeg, vectors["mixamorig:RightLeg"]);
        }
        catch (System.Exception)
        {
            // Silenciar para no inundar la consola si llega un paquete malformado en UDP
        }
    }

    void ApplyBoneTransform(Transform bone, JToken data)
    {
        if (bone != null && data != null && initialRotations.ContainsKey(bone) && initialDirections.ContainsKey(bone))
        {
            // Los vectores de Python ya vienen con los ejes adaptados a Unity (Y arriba, Z frente)
            float x = (float)data["x"];
            float y = (float)data["y"];
            float z = (float)data["z"];
            
            // Eliminamos la inversión de X (mirrorMode) porque MediaPipe ya nos da un
            // sistema de coordenadas nativamente espejeado cuando miramos a la webcam.
            // Invertir la X causaba que el hueso se "atorara" apuntando hacia afuera
            // y no pudiera cruzar el pecho del personaje.
            
            Vector3 targetDir = new Vector3(x, y, z).normalized;
            Vector3 initialDir = initialDirections[bone];
            
            // Si hay un vector valido, calculamos la rotacion
            if (targetDir.sqrMagnitude > 0)
            {
                // Rotacion desde la direccion T-Pose hacia la nueva direccion
                Quaternion rot = Quaternion.FromToRotation(initialDir, targetDir);
                
                // Aplicamos rotacion al hueso original usando Slerp para suavizar el movimiento
                Quaternion targetRotation = rot * initialRotations[bone];
                bone.rotation = Quaternion.Slerp(bone.rotation, targetRotation, Time.deltaTime * 25f);
            }
        }
    }
}
