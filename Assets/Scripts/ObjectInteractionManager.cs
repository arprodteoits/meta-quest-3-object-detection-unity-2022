using UnityEngine;
using System.Collections.Generic;

public class ObjectInteractionManager : MonoBehaviour
{
    // Struktur data untuk menampung info objek
    public struct DetectedObject
    {
        public string label;
        public Vector2 centroid; // Koordinat tengah (x, y)
    }

    [Header("Settings")]
    public float interactionThreshold = 0.15f; // Jarak untuk dianggap "dekat"

    private List<DetectedObject> currentDetections = new List<DetectedObject>();

    // Fungsi ini dipanggil dari YoloWebcamDemo
    public void UpdateDetections(List<DetectedObject> newDetections)
    {
        currentDetections = newDetections;
        CheckInteractions();
    }

    void CheckInteractions()
    {
        // Butuh minimal 2 benda untuk menghitung jarak
        if (currentDetections.Count < 2) return;

        for (int i = 0; i < currentDetections.Count; i++)
        {
            for (int j = i + 1; j < currentDetections.Count; j++)
            {
                // Menghitung Euclidean Distance (rumus d = sqrt((x2-x1)^2 + (y2-y1)^2))
                float distance = Vector2.Distance(currentDetections[i].centroid, currentDetections[j].centroid);

                if (distance < interactionThreshold)
                {
                    Debug.Log($"<color=red>INTERAKSI!</color> {currentDetections[i].label} dekat dengan {currentDetections[j].label}. Jarak: {distance:F2}");
                }
            }
        }
    }
}