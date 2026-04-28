using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;
using UnityEngine.UI;

public class YoloWebcamDemo : MonoBehaviour
{
    [Header("AI & GPU Settings")]
    public ModelAsset modelAsset;
    public ComputeShader yuvToRgbShader; // Tempat memasukkan YUVToRGBShader kamu
    private Model runtimeModel;
    private Worker worker;
    private Tensor<float> inputTensor;

    [Header("UI & Visuals")]
    public RectTransform displayArea; 
    public GameObject boxPrefab; 
    private List<GameObject> activeBoxes = new List<GameObject>();

    [Header("Performance Throttling")]
    public float inferenceInterval = 0.2f; // 5 FPS
    private float timer = 0f;

    [Header("YOLO Parameters")]
    [Range(0.0f, 1.0f)] public float confidenceThreshold = 0.35f;
    [Range(0.0f, 1.0f)] public float smoothingFactor = 0.85f; 

    private const int NUM_PROPOSALS = 8400; 
    private const int IMAGE_SIZE = 640;
    
    // Tekstur RGB hasil konversi GPU
    private RenderTexture rgbRenderTexture;
    private int kernelIndex;

    private BoundingBox previousSmoothedBox;
    private bool hasPreviousBox = false;

    public struct BoundingBox { public float cx, cy, w, h, conf; }

    void Start()
    {
        // 1. Inisialisasi Otak Sentis
        if (modelAsset != null)
        {
            runtimeModel = ModelLoader.Load(modelAsset);
            worker = new Worker(runtimeModel, BackendType.GPUCompute);
            inputTensor = new Tensor<float>(new TensorShape(1, 3, IMAGE_SIZE, IMAGE_SIZE));
        }

        // 2. Siapkan "Kanvas Kosong" (RenderTexture) untuk hasil Compute Shader
        // Ukuran 640x640 disesuaikan langsung dengan mulut YOLO
        rgbRenderTexture = new RenderTexture(IMAGE_SIZE, IMAGE_SIZE, 0, RenderTextureFormat.ARGB32);
        rgbRenderTexture.enableRandomWrite = true; // Wajib agar Compute Shader bisa menggambar di sini
        rgbRenderTexture.Create();

        if (yuvToRgbShader != null)
        {
            kernelIndex = yuvToRgbShader.FindKernel("CSMain");
        }
    }

    void Update()
    {
        timer += Time.deltaTime;

        // Hanya jalankan AI setiap beberapa milidetik (Throttling)
        if (timer >= inferenceInterval)
        {
            ProcessCameraFrame();
            timer = 0f;
        }
    }

    void ProcessCameraFrame()
    {
        // --- BLOK PENYEDOT KAMERA META QUEST ---
        // Karena ini adalah prototipe, kita asumsikan OVRManager sudah memberikan akses.
        // Di aplikasi penuh, di sini kita memanggil OVRPlugin.GetPassthroughCameraFrame()
        // Namun, untuk menghindari error kompilasi karena perbedaan versi SDK, 
        // kita akan melakukan konversi gambar yang terlihat di layar secara aman via GPU.

        Texture2D currentScreen = ScreenCapture.CaptureScreenshotAsTexture();
        if (currentScreen == null) return;

        // Jalankan Compute Shader untuk membersihkan dan menyesuaikan gambar
        if (yuvToRgbShader != null)
        {
            yuvToRgbShader.SetTexture(kernelIndex, "YTex", currentScreen); // Simulasi input
            yuvToRgbShader.SetTexture(kernelIndex, "UVTex", currentScreen); // Simulasi input
            yuvToRgbShader.SetTexture(kernelIndex, "Result", rgbRenderTexture);
            
            // Bagi tugas ke GPU (640/8 = 80 blok kerja)
            yuvToRgbShader.Dispatch(kernelIndex, IMAGE_SIZE / 8, IMAGE_SIZE / 8, 1);
        }

        // Suapkan hasil konversi GPU ke Sentis AI
        ExecuteInference(rgbRenderTexture);
        
        Destroy(currentScreen); // Cegah memori bocor (Memory Leak)
    }

    void ExecuteInference(RenderTexture sourceTexture)
    {
        if (worker == null || inputTensor == null) return;

        TextureTransform transform = new TextureTransform().SetDimensions(IMAGE_SIZE, IMAGE_SIZE).SetTensorLayout(TensorLayout.NCHW);
        TextureConverter.ToTensor(sourceTexture, inputTensor, transform);
        worker.Schedule(inputTensor);

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
        return interArea / (boxA.w * boxA.h + boxB.w * boxB.h - interArea);
    }

    private void OnDisable()
    {
        worker?.Dispose();
        inputTensor?.Dispose();
        if (rgbRenderTexture != null) rgbRenderTexture.Release();
    }
}