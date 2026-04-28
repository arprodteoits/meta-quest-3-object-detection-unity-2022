using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;
using UnityEngine.UI;

public class YoloQuestStandalone : MonoBehaviour
{
    [Header("Sentis AI Settings")]
    public ModelAsset modelAsset;
    private Model runtimeModel;
    private Worker worker;
    private Tensor<float> inputTensor;

    [Header("UI & Visuals")]
    public RectTransform displayArea; // Canvas atau panel tempat menggambar kotak
    public GameObject boxPrefab; 
    private List<GameObject> activeBoxes = new List<GameObject>();

    [Header("Performance Optimization")]
    [Tooltip("Berapa detik sekali AI menebak? 0.2 = 5 FPS (Sangat aman untuk Quest 3)")]
    public float inferenceInterval = 0.2f; 
    private float timer = 0f;

    [Header("YOLO Settings")]
    [Range(0.0f, 1.0f)]
    public float confidenceThreshold = 0.35f;
    [Range(0.0f, 1.0f)]
    public float smoothingFactor = 0.85f; 

    private const int NUM_PROPOSALS = 8400; 
    private const int IMAGE_SIZE = 640;

    // Tekstur untuk menampung tangkapan layar Quest
    private Texture2D screenCaptureTexture;
    
    private BoundingBox previousSmoothedBox;
    private bool hasPreviousBox = false;

    public struct BoundingBox
    {
        public float cx, cy, w, h, conf;
    }

    void Start()
    {
        // 1. Setup AI Sentis
        if (modelAsset != null)
        {
            runtimeModel = ModelLoader.Load(modelAsset);
            worker = new Worker(runtimeModel, BackendType.GPUCompute);
            inputTensor = new Tensor<float>(new TensorShape(1, 3, IMAGE_SIZE, IMAGE_SIZE));
            Debug.Log("Otak YOLO Siap!");
        }

        // 2. Siapkan kanvas kosong untuk foto layar (sesuai resolusi mata Quest)
        screenCaptureTexture = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);

        // 3. Mulai siklus pemotretan layar
        StartCoroutine(CaptureAndAnalyzeFrame());
    }

    private IEnumerator CaptureAndAnalyzeFrame()
    {
        while (true)
        {
            // Tunggu sampai semua proses render grafik Unity (dan Passthrough) selesai di frame ini
            yield return new WaitForEndOfFrame();

            timer += Time.deltaTime;

            // Jalankan AI hanya jika waktunya sudah tiba (Throttling agar tidak crash)
            if (timer >= inferenceInterval)
            {
                // "Foto" layar yang dilihat user
                screenCaptureTexture.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
                screenCaptureTexture.Apply();

                // Kirim foto ke YOLO
                ExecuteInference(screenCaptureTexture);
                
                timer = 0f; // Reset timer
            }
        }
    }

    void ExecuteInference(Texture2D sourceTexture)
    {
        if (worker == null || inputTensor == null) return;

        // Ubah foto layar menjadi format yang dipahami AI (640x640)
        TextureTransform transform = new TextureTransform().SetDimensions(IMAGE_SIZE, IMAGE_SIZE).SetTensorLayout(TensorLayout.NCHW);
        TextureConverter.ToTensor(sourceTexture, inputTensor, transform);
        worker.Schedule(inputTensor);

        // Ambil hasil tebakan dari GPU
        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
        
        if (outputTensor != null)
        {
            float[] data = outputTensor.DownloadToArray();
            ParseYOLOOutput(data);
        }
    }

    void ParseYOLOOutput(float[] data)
    {
        List<BoundingBox> boxes = new List<BoundingBox>();

        for (int i = 0; i < NUM_PROPOSALS; i++)
        {
            float conf = data[4 * NUM_PROPOSALS + i];
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

        boxes.Sort((a, b) => b.conf.CompareTo(a.conf));
        List<BoundingBox> finalBoxes = new List<BoundingBox>();

        // Non-Maximum Suppression (NMS)
        foreach (var box in boxes)
        {
            bool isOverlapping = false;
            foreach (var finalBox in finalBoxes)
            {
                if (CalculateIoU(box, finalBox) > 0.45f)
                {
                    isOverlapping = true;
                    break;
                }
            }
            if (!isOverlapping) finalBoxes.Add(box);
        }

        if (finalBoxes.Count > 0)
        {
            BoundingBox currentRawBox = finalBoxes[0];

            if (!hasPreviousBox)
            {
                previousSmoothedBox = currentRawBox;
                hasPreviousBox = true;
            }
            else
            {
                // Temporal Smoothing agar kotak tidak bergetar (jitter)
                previousSmoothedBox.cx = Mathf.Lerp(currentRawBox.cx, previousSmoothedBox.cx, smoothingFactor);
                previousSmoothedBox.cy = Mathf.Lerp(currentRawBox.cy, previousSmoothedBox.cy, smoothingFactor);
                previousSmoothedBox.w = Mathf.Lerp(currentRawBox.w, previousSmoothedBox.w, smoothingFactor);
                previousSmoothedBox.h = Mathf.Lerp(currentRawBox.h, previousSmoothedBox.h, smoothingFactor);
            }

            DrawBoxes(new List<BoundingBox> { previousSmoothedBox });
        }
        else
        {
            DrawBoxes(new List<BoundingBox>());
            hasPreviousBox = false;
        }
    }

    void DrawBoxes(List<BoundingBox> boxesToDraw)
    {
        foreach (var boxObj in activeBoxes) Destroy(boxObj);
        activeBoxes.Clear();

        if (displayArea == null) return;

        Vector2 uiSize = displayArea.rect.size;

        foreach (var box in boxesToDraw)
        {
            GameObject newBox = Instantiate(boxPrefab, displayArea);
            RectTransform rt = newBox.GetComponent<RectTransform>();

            rt.anchorMin = new Vector2(0.5f, 0.5f);
            rt.anchorMax = new Vector2(0.5f, 0.5f);
            rt.pivot = new Vector2(0.5f, 0.5f);

            // Konversi koordinat AI (0-1) ke koordinat UI (Pixel)
            float xNorm = box.cx / IMAGE_SIZE;
            float yNorm = box.cy / IMAGE_SIZE;
            float wNorm = box.w / IMAGE_SIZE;
            float hNorm = box.h / IMAGE_SIZE;

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
        if (screenCaptureTexture != null) Destroy(screenCaptureTexture);
    }
}