using System.Collections.Generic;
using UnityEngine;

public class ObjectInteractionManager : MonoBehaviour
{
    // Struktur data yang dikirim dari YoloWebcamDemo
    public struct DetectedObject
    {
        public string label;
        public Vector2 centroid;
    }

    [Header("Pengaturan Interaksi")]
    [Tooltip("Jarak maksimal (dalam pixel) agar interaksi terjadi")]
    public float interactionThreshold = 150f; 

    [Header("UI Shapes (Masukkan Prefab / UI Image)")]
    public GameObject circleShape;   // Muncul jika Bottle + Pen
    public GameObject triangleShape; // Muncul jika Bottle + Eraser
    public GameObject squareShape;   // Muncul jika Pen + Eraser

    // Fungsi ini dipanggil setiap frame oleh YoloWebcamDemo
    public void UpdateDetections(List<DetectedObject> detections)
    {
        // 1. Matikan (Sembunyikan) semua bentuk di awal frame
        if (circleShape != null) circleShape.SetActive(false);
        if (triangleShape != null) triangleShape.SetActive(false);
        if (squareShape != null) squareShape.SetActive(false);

        // Jika benda yang terdeteksi kurang dari 2, hentikan proses (tidak mungkin ada jarak antar 2 benda)
        if (detections.Count < 2) return;

        // 2. Cari benda-benda spesifik di dalam daftar deteksi
        DetectedObject bottle = default;
        DetectedObject pen = default;
        DetectedObject eraser = default;

        bool hasBottle = false, hasPen = false, hasEraser = false;

        foreach (var obj in detections)
        {
            if (obj.label == "bottle") { bottle = obj; hasBottle = true; }
            else if (obj.label == "pen") { pen = obj; hasPen = true; }
            else if (obj.label == "eraser") { eraser = obj; hasEraser = true; }
        }

        // 3. --- LOGIKA EUCLIDEAN DISTANCE ---

        // Kombinasi A: Bottle & Pen -> Lingkaran
        if (hasBottle && hasPen)
        {
            // Vector2.Distance otomatis menggunakan rumus Euclidean Distance: akar( (x2-x1)^2 + (y2-y1)^2 )
            float distance = Vector2.Distance(bottle.centroid, pen.centroid);
            
            if (distance <= interactionThreshold && circleShape != null)
            {
                circleShape.SetActive(true);
                Debug.Log($"Interaksi: Bottle & Pen berdekatan! Jarak: {distance:0.0} px");
            }
        }

        // Kombinasi B: Bottle & Eraser -> Segitiga
        if (hasBottle && hasEraser)
        {
            float distance = Vector2.Distance(bottle.centroid, eraser.centroid);
            
            if (distance <= interactionThreshold && triangleShape != null)
            {
                triangleShape.SetActive(true);
                Debug.Log($"Interaksi: Bottle & Eraser berdekatan! Jarak: {distance:0.0} px");
            }
        }

        // Kombinasi C: Pen & Eraser -> Kotak
        if (hasPen && hasEraser)
        {
            float distance = Vector2.Distance(pen.centroid, eraser.centroid);
            
            if (distance <= interactionThreshold && squareShape != null)
            {
                squareShape.SetActive(true);
                Debug.Log($"Interaksi: Pen & Eraser berdekatan! Jarak: {distance:0.0} px");
            }
        }
    }
}