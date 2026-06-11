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
    public GameObject circleShape;   // Muncul jika Bottle + Eraser
    public GameObject triangleShape; // Muncul jika Bottle + Pen
    public GameObject squareShape;   // Muncul jika Eraser + Pen



    void Start()
    {
        // Otomatis menyembunyikan semua bentuk saat game pertama kali Play
        if (circleShape != null) circleShape.SetActive(false);
        if (triangleShape != null) triangleShape.SetActive(false);
        if (squareShape != null) squareShape.SetActive(false);
    }

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
        DetectedObject eraser = default;
        DetectedObject pen = default;

        bool hasBottle = false, hasEraser = false, hasPen = false;

        foreach (var obj in detections)
        {
            if (obj.label == "bottle") { bottle = obj; hasBottle = true; }
            else if (obj.label == "eraser") { eraser = obj; hasEraser = true; }
            else if (obj.label == "pen") { pen = obj; hasPen = true; }
        }

        // 3. --- LOGIKA EUCLIDEAN DISTANCE ---

        // Kombinasi A: Bottle & Eraser -> Lingkaran
        if (hasBottle && hasEraser)
        {
            // Vector2.Distance otomatis menggunakan rumus Euclidean Distance: akar( (x2-x1)^2 + (y2-y1)^2 )
            float distance = Vector2.Distance(bottle.centroid, eraser.centroid);
            
            if (distance <= interactionThreshold && circleShape != null)
            {
                circleShape.SetActive(true);
                Debug.Log($"Interaksi: Bottle & Eraser berdekatan! Jarak: {distance:0.0} px");
            }
        }

        // Kombinasi B: Bottle & Pen -> Segitiga
        if (hasBottle && hasPen)
        {
            float distance = Vector2.Distance(bottle.centroid, pen.centroid);
            
            if (distance <= interactionThreshold && triangleShape != null)
            {
                triangleShape.SetActive(true);
                Debug.Log($"Interaksi: Bottle & Pen berdekatan! Jarak: {distance:0.0} px");
            }
        }

        // Kombinasi C: Eraser & Pen -> Kotak
        if (hasEraser && hasPen)
        {
            float distance = Vector2.Distance(eraser.centroid, pen.centroid);
            
            if (distance <= interactionThreshold && squareShape != null)
            {
                squareShape.SetActive(true);
                Debug.Log($"Interaksi: Eraser & Pen berdekatan! Jarak: {distance:0.0} px");
            }
        }
    }
}