using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis; 
using UnityEngine.UI;
using TMPro;

public class YoloWebcamDemo : MonoBehaviour
{
    [Header("Sentis AI")]
    public ModelAsset modelAsset;
    private Model runtimeModel;
    private Worker worker; 
    private Tensor<float> inputTensor;

    // --- DAFTAR KELAS SESUAI ROBOFLOW ---
    private string[] classNames = { "bottle", "eraser", "pen" }; 

    [Header("Kamera & UI")]
    public int webcamIndex = 1; // 0 biasanya internal, 1 biasanya eksternal
    public RawImage displayImage;
    public GameObject boxPrefab; 
    public ObjectInteractionManager interactionManager; // Hubungkan ke Manager Jarak
    
    private WebCamTexture webcamTexture;
    private List<GameObject> activeBoxes = new List<GameObject>(); 

    private BoundingBox previousSmoothedBox;
    private bool hasPreviousBox = false;

    [Range(0.0f, 1.0f)]
    public float smoothingFactor = 0.85f; 
    [Range(0.0f, 1.0f)]
    public float confidenceThreshold = 0.35f;

    private const int NUM_PROPOSALS = 8400; 
    private const int IMAGE_SIZE = 640;

    public struct BoundingBox {
        public float cx, cy, w, h, conf;
        public string label; // Tambahkan label untuk membedakan benda
    }

    void Start() {
        //  webcamTexture = new WebCamTexture();
        // if (displayImage != null) displayImage.texture = webcamTexture;
        // webcamTexture.Play();

    // void Start() {
      WebCamDevice[] devices = WebCamTexture.devices;
      string selectedCameraName = "";

    Debug.Log("--- Mencari Kamera Eksternal ---");
    for (int i = 0; i < devices.Length; i++) {
        Debug.Log($"Ditemukan Indeks [{i}]: {devices[i].name}");

        // Cari yang namanya mengandung "Logitech" (tidak peduli huruf besar/kecil)
        if (devices[i].name.ToLower().Contains("logitech")) {
            selectedCameraName = devices[i].name;
            Debug.Log("Kamera Logitech Ditemukan! Menggunakan: " + selectedCameraName);
            break; 
        }
    }

    // Jika Logitech tidak ketemu, cari yang BUKAN Lenovo
    if (string.IsNullOrEmpty(selectedCameraName)) {
        foreach (var dev in devices) {
            if (!dev.name.ToLower().Contains("lenovo") && !dev.name.ToLower().Contains("easycamera")) {
                selectedCameraName = dev.name;
                break;
            }
        }
    }

    // Eksekusi Kamera
    if (!string.IsNullOrEmpty(selectedCameraName)) {
        webcamTexture = new WebCamTexture(selectedCameraName, IMAGE_SIZE, IMAGE_SIZE);
    } else {
        // Fallback terakhir kalau semua gagal
        webcamTexture = new WebCamTexture(devices[0].name);
        Debug.LogWarning("Logitech tidak ketemu, terpaksa pakai kamera default.");
    }

    if (displayImage != null) displayImage.texture = webcamTexture;
    webcamTexture.Play();

                            // ... (sisanya tetap sama untuk loading model AI)

        if (modelAsset != null) {
            runtimeModel = ModelLoader.Load(modelAsset);
            worker = new Worker(runtimeModel, BackendType.GPUCompute);
            inputTensor = new Tensor<float>(new TensorShape(1, 3, IMAGE_SIZE, IMAGE_SIZE));
        }
    }

    void Update() {
        if (webcamTexture != null && webcamTexture.didUpdateThisFrame) ExecuteInference();
    }

    void ExecuteInference() {
        TextureTransform transform = new TextureTransform().SetDimensions(IMAGE_SIZE, IMAGE_SIZE).SetTensorLayout(TensorLayout.NCHW);
        TextureConverter.ToTensor(webcamTexture, inputTensor, transform);
        worker.Schedule(inputTensor);

        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
        if (outputTensor != null) {
            float[] data = outputTensor.DownloadToArray();
            ParseYOLOOutput(data);
        }
    }

    void ParseYOLOOutput(float[] data) {
        List<BoundingBox> boxes = new List<BoundingBox>();
        int numClasses = classNames.Length;

        for (int i = 0; i < NUM_PROPOSALS; i++) {
            // 1. Cari skor tertinggi di antara 3 kelas
            float maxScore = 0;
            int bestClassId = 0;

            for (int c = 0; c < numClasses; c++) {
                // Skor kelas dimulai dari indeks ke-4, 5, dan 6
                float score = data[(4 + c) * NUM_PROPOSALS + i];
                if (score > maxScore) {
                    maxScore = score;
                    bestClassId = c;
                }
            }

            // 2. Filter dengan Threshold
            if (maxScore > confidenceThreshold) {
                boxes.Add(new BoundingBox {
                    cx = data[0 * NUM_PROPOSALS + i],
                    cy = data[1 * NUM_PROPOSALS + i],
                    w = data[2 * NUM_PROPOSALS + i],
                    h = data[3 * NUM_PROPOSALS + i],
                    conf = maxScore,
                    label = classNames[bestClassId] // Simpan nama benda
                });
            }
        }

        // 3. NMS (Sederhana)
        boxes.Sort((a, b) => b.conf.CompareTo(a.conf));
        List<BoundingBox> finalBoxes = new List<BoundingBox>();
        foreach (var box in boxes) {
            bool overlap = false;
            foreach (var fb in finalBoxes) {
                if (CalculateIoU(box, fb) > 0.45f) { overlap = true; break; }
            }
            if (!overlap) finalBoxes.Add(box);
        }

        // 4. Kirim data ke Interaction Manager untuk hitung jarak
        // 4. Kirim data ke Interaction Manager untuk hitung jarak
        if (interactionManager != null) {
            // Ambil ukuran layar UI kamera saat ini
            Vector2 uiSize = displayImage.rectTransform.rect.size; 
            List<ObjectInteractionManager.DetectedObject> interactionList = new List<ObjectInteractionManager.DetectedObject>();
            
            foreach (var b in finalBoxes) {
                // Terjemahkan koordinat YOLO (640) menjadi koordinat Layar UI
                float xNorm = b.cx / IMAGE_SIZE;
                float yNorm = b.cy / IMAGE_SIZE;
                float uiX = (xNorm * uiSize.x) - (uiSize.x / 2f);
                float uiY = (uiSize.y / 2f) - (yNorm * uiSize.y); 

                interactionList.Add(new ObjectInteractionManager.DetectedObject {
                    label = b.label,
                    centroid = new Vector2(uiX, uiY) // <- Sekarang titik tengah sesuai dengan layar
                });
            }
            interactionManager.UpdateDetections(interactionList);
        }

        DrawBoxes(finalBoxes);
    }

    // --- Fungsi Helper (DrawBoxes, CalculateIoU, OnDisable) tetap sama seperti sebelumnya ---
    void DrawBoxes(List<BoundingBox> boxesToDraw) {
        foreach (var boxObj in activeBoxes) Destroy(boxObj);
        activeBoxes.Clear();
        Vector2 uiSize = displayImage.rectTransform.rect.size;

        foreach (var box in boxesToDraw) {
            GameObject newBox = Instantiate(boxPrefab, displayImage.transform);
            RectTransform rt = newBox.GetComponent<RectTransform>();
            rt.anchorMin = rt.anchorMax = rt.pivot = new Vector2(0.5f, 0.5f);

            float xNorm = box.cx / IMAGE_SIZE;
            float yNorm = box.cy / IMAGE_SIZE;
            
            float uiX = (xNorm * uiSize.x) - (uiSize.x / 2f);
            float uiY = (uiSize.y / 2f) - (yNorm * uiSize.y); 

            rt.anchoredPosition = new Vector2(uiX, uiY);
            rt.sizeDelta = new Vector2((box.w / IMAGE_SIZE) * uiSize.x, (box.h / IMAGE_SIZE) * uiSize.y);
            
            // Tampilkan nama benda di console (opsional)
            // Debug.Log("Terdeteksi: " + box.label);

                                 // TAMBAHKAN KODE INI:
                // Cari komponen teks di dalam prefab, lalu isi dengan nama label
                var textComponent = newBox.GetComponentInChildren<TextMeshProUGUI>();
                if (textComponent != null) {
                    // Mengisi teks dengan "Nama Benda" + "Skor Akurasi"
                    textComponent.text = $"{box.label} {(box.conf * 100):0}%";
        }


            activeBoxes.Add(newBox);

            
        }
    }

    float CalculateIoU(BoundingBox boxA, BoundingBox boxB) {
        float xA = Mathf.Max(boxA.cx - boxA.w / 2, boxB.cx - boxB.w / 2);
        float yA = Mathf.Max(boxA.cy - boxA.h / 2, boxB.cy - boxB.h / 2);
        float xB = Mathf.Min(boxA.cx + boxA.w / 2, boxB.cx + boxB.w / 2);
        float yB = Mathf.Min(boxA.cy + boxA.h / 2, boxB.cy + boxB.h / 2);
        float interArea = Mathf.Max(0, xB - xA) * Mathf.Max(0, yB - yA);
        return interArea / (boxA.w * boxA.h + boxB.w * boxB.h - interArea);
    }

    private void OnDisable() {
        worker?.Dispose();
        inputTensor?.Dispose();
        if (webcamTexture != null) webcamTexture.Stop();
    }
}