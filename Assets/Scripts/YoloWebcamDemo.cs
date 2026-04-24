using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis; // Ganti using Unity.Sentis.Layers; ke using Unity.Sentis; untuk versi 2.1.3
using UnityEngine.UI;

public class YoloWebcamDemo : MonoBehaviour
{
    [Header("Sentis AI")]
    public ModelAsset modelAsset;
    private Model runtimeModel;
    private Worker worker; // Sentis 2.1.3 menggunakan Worker (tanpa 'I')
    private Tensor<float> inputTensor;

    [Header("Kamera & UI")]
    public RawImage displayImage;
    public GameObject boxPrefab; 
    
    private WebCamTexture webcamTexture;
    private List<GameObject> activeBoxes = new List<GameObject>(); // Menyimpan kotak yang tampil

    // --- TEMPORAL SMOOTHING VARIABLES ---
    // Kita simpan posisi 'rata-rata' kotak sebelumnya
    private BoundingBox previousSmoothedBox;
    private bool hasPreviousBox = false;

    // Seberapa mulus pergerakan kotak (0.0f = tidak mulus sama sekali, 1.0f = kotak tidak bergerak)
    // Coba ganti nilainya antara 0.6f - 0.9f untuk melihat perbedaan
    [Range(0.0f, 1.0f)]
    public float smoothingFactor = 0.85f; 
    
    // Threshold confidence AI, turunkan untuk membuat AI lebih 'pemaaf'
    [Range(0.0f, 1.0f)]
    public float confidenceThreshold = 0.35f;

    // Total tebakan YOLO
    private const int NUM_PROPOSALS = 8400; 
    private const int IMAGE_SIZE = 640;

    // Struktur data untuk kotak
    public struct BoundingBox
    {
        public float cx, cy, w, h, conf;
    }

    void Start()
    {
        // 1. Setup Kamera
        webcamTexture = new WebCamTexture();
        if (displayImage != null) displayImage.texture = webcamTexture;
        webcamTexture.Play();

        // 2. Setup Sentis (Sintaks versi 2.1.3)
        if (modelAsset != null)
        {
            runtimeModel = ModelLoader.Load(modelAsset);
            // new Worker(...) untuk membuat worker di Sentis 2.1.3
            worker = new Worker(runtimeModel, BackendType.GPUCompute);
            inputTensor = new Tensor<float>(new TensorShape(1, 3, IMAGE_SIZE, IMAGE_SIZE));
            Debug.Log("Selamat! Sistem AI & Temporal Smoothing Siap!");
        }
    }

    void Update()
    {
        if (webcamTexture != null && webcamTexture.didUpdateThisFrame)
        {
            ExecuteInference();
        }
    }

    void ExecuteInference()
    {
        TextureTransform transform = new TextureTransform().SetDimensions(IMAGE_SIZE, IMAGE_SIZE).SetTensorLayout(TensorLayout.NCHW);
        TextureConverter.ToTensor(webcamTexture, inputTensor, transform);
        worker.Schedule(inputTensor); // Sentis 2.1.3 menggunakan .Schedule()

        // Ambil hasil tensor
        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
        
        if (outputTensor != null)
        {
            // --- FIXED DownloadToArray() ---
            // .DownloadToArray() untuk Sentis 2.1.3 agar data turun dari GPU ke CPU
            float[] data = outputTensor.DownloadToArray();
            ParseYOLOOutput(data);
        }
    }

    void ParseYOLOOutput(float[] data)
    {
        List<BoundingBox> boxes = new List<BoundingBox>();

        // 1. FILTERING: Loop semua tebakan
        for (int i = 0; i < NUM_PROPOSALS; i++)
        {
            float conf = data[4 * NUM_PROPOSALS + i];
            
            // Gunakan Threshold yang lebih pemaaf agar kotak tidak mudah hilang
            if (conf > confidenceThreshold)
            {
                BoundingBox box = new BoundingBox
                {
                    cx = data[0 * NUM_PROPOSALS + i],
                    cy = data[1 * NUM_PROPOSALS + i],
                    w = data[2 * NUM_PROPOSALS + i],
                    h = data[3 * NUM_PROPOSALS + i],
                    conf = conf
                };
                boxes.Add(box);
            }
        }

        // 2. NON-MAXIMUM SUPPRESSION (NMS): Hapus kotak yang numpuk
        boxes.Sort((a, b) => b.conf.CompareTo(a.conf)); // Urutkan dari yang paling yakin
        List<BoundingBox> finalBoxes = new List<BoundingBox>();

        foreach (var box in boxes)
        {
            bool isOverlapping = false;
            foreach (var finalBox in finalBoxes)
            {
                if (CalculateIoU(box, finalBox) > 0.45f) // Batas tumpukan 45%
                {
                    isOverlapping = true;
                    break;
                }
            }
            if (!isOverlapping) finalBoxes.Add(box);
        }

        // --- APPLY TEMPORAL SMOOTHING ---
        // Jika kita berhasil mendeteksi setidaknya satu kotak
        if (finalBoxes.Count > 0)
        {
            // Ambil tebakan mentah terbaik dari YOLO
            BoundingBox currentRawBox = finalBoxes[0];

            // Jika sebelumnya kita belum punya data smoothing (frame pertama)
            if (!hasPreviousBox)
            {
                previousSmoothedBox = currentRawBox;
                hasPreviousBox = true;
            }
            else
            {
                // **Rumus Smoothing Sederhana**
                // Rata-rata tertimbang antara posisi lama dan posisi baru
                previousSmoothedBox.cx = Mathf.Lerp(currentRawBox.cx, previousSmoothedBox.cx, smoothingFactor);
                previousSmoothedBox.cy = Mathf.Lerp(currentRawBox.cy, previousSmoothedBox.cy, smoothingFactor);
                previousSmoothedBox.w = Mathf.Lerp(currentRawBox.w, previousSmoothedBox.w, smoothingFactor);
                previousSmoothedBox.h = Mathf.Lerp(currentRawBox.h, previousSmoothedBox.h, smoothingFactor);
                // previousSmoothedBox.conf = currentRawBox.conf;
            }

            // Gunakan kotak rata-rata ini untuk menggambar di layar
            List<BoundingBox> smoothedBoxes = new List<BoundingBox>();
            smoothedBoxes.Add(previousSmoothedBox);
            DrawBoxes(smoothedBoxes);
        }
        else
        {
            // Jika tidak ada deteksi, hapus kotak (opsional, bisa juga kita biarkan bertahan sejenak)
            DrawBoxes(new List<BoundingBox>());
            hasPreviousBox = false; // Reset smoothing jika objek hilang
        }
    }

    void DrawBoxes(List<BoundingBox> boxesToDraw)
    {
        // Hapus kotak frame sebelumnya
        foreach (var boxObj in activeBoxes) Destroy(boxObj);
        activeBoxes.Clear();

        Vector2 uiSize = displayImage.rectTransform.rect.size;

        foreach (var box in boxesToDraw)
        {
            GameObject newBox = Instantiate(boxPrefab, displayImage.transform);
            RectTransform rt = newBox.GetComponent<RectTransform>();

            // Paksa Anchor & Pivot ke tengah (0.5, 0.5) untuk mencegah kotak melar
            rt.anchorMin = new Vector2(0.5f, 0.5f);
            rt.anchorMax = new Vector2(0.5f, 0.5f);
            rt.pivot = new Vector2(0.5f, 0.5f);

            // Normalisasi koordinat YOLO (640x640)
            float xNorm = box.cx / IMAGE_SIZE;
            float yNorm = box.cy / IMAGE_SIZE;
            float wNorm = box.w / IMAGE_SIZE;
            float hNorm = box.h / IMAGE_SIZE;

            // Konversi ke ukuran layar UI
            float uiX = (xNorm * uiSize.x) - (uiSize.x / 2f);
            float uiY = (uiSize.y / 2f) - (yNorm * uiSize.y); 

            rt.anchoredPosition = new Vector2(uiX, uiY);
            rt.sizeDelta = new Vector2(wNorm * uiSize.x, hNorm * uiSize.y);

            activeBoxes.Add(newBox);
        }
    }

    float CalculateIoU(BoundingBox boxA, BoundingBox boxB)
    {
        float xA = Mathf.Max(boxA.cx - boxA.w / 2, boxB.cx - boxB.w / 2);
        float yA = Mathf.Max(boxA.cy - boxA.h / 2, boxB.cy - boxB.h / 2);
        float xB = Mathf.Min(boxA.cx + boxA.w / 2, boxB.cx + boxB.w / 2);
        float yB = Mathf.Min(boxA.cy + boxA.h / 2, boxB.cy + boxB.h / 2);

        float interArea = Mathf.Max(0, xB - xA) * Mathf.Max(0, yB - yA);
        float boxAArea = boxA.w * boxA.h;
        float boxBArea = boxB.w * boxB.h;

        return interArea / (boxAArea + boxBArea - interArea);
    }

    private void OnDisable()
    {
        worker?.Dispose();
        inputTensor?.Dispose();
        if (webcamTexture != null) webcamTexture.Stop();
    }
}