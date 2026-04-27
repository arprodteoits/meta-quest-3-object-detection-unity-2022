using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;
using UnityEngine.UI;
using UnityEngine.Android; // WAJIB UNTUK QUEST 3

public class YoloWebcamDemo : MonoBehaviour
{
    [Header("Sentis AI")]
    public ModelAsset modelAsset;
    private Model runtimeModel;
    private Worker worker;
    private Tensor<float> inputTensor;

    [Header("Kamera & UI")]
    public RawImage displayImage;
    public GameObject boxPrefab; 
    
    private WebCamTexture webcamTexture;
    private List<GameObject> activeBoxes = new List<GameObject>();

    // Variabel Temporal Smoothing
    private BoundingBox previousSmoothedBox;
    private bool hasPreviousBox = false;

    [Range(0.0f, 1.0f)]
    public float smoothingFactor = 0.85f; 
    [Range(0.0f, 1.0f)]
    public float confidenceThreshold = 0.35f;

    private const int NUM_PROPOSALS = 8400; 
    private const int IMAGE_SIZE = 640;

    public struct BoundingBox
    {
        public float cx, cy, w, h, conf;
    }

    void Start()
    {
        // 1. CEK IZIN KAMERA DULU (Sistem Keamanan Meta Quest)
        if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            Debug.Log("Meminta izin kamera ke OS Meta Quest...");
            Permission.RequestUserPermission(Permission.Camera);
            // Tunggu user klik "Allow" sebelum menyalakan AI
            StartCoroutine(WaitForCameraPermission());
        }
        else
        {
            // Jika sudah diizinkan sebelumnya, langsung gas!
            InitializeSystem();
        }
    }

    // Coroutine untuk menunggu pengguna mengklik tombol "Allow" di VR
    private IEnumerator WaitForCameraPermission()
    {
        while (!Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            yield return new WaitForSeconds(0.5f);
        }
        InitializeSystem();
    }

    void InitializeSystem()
    {
        // Setup Kamera setelah izin didapat
        // (Di Quest 3, kita ambil kamera pertama yang terdeteksi)
        if (WebCamTexture.devices.Length > 0)
        {
            webcamTexture = new WebCamTexture(WebCamTexture.devices[0].name, 1280, 720, 30);
            if (displayImage != null) displayImage.texture = webcamTexture;
            webcamTexture.Play();
        }
        else
        {
            Debug.LogError("Tidak ada kamera yang terdeteksi di perangkat ini!");
            return;
        }

        // Setup AI Sentis
        if (modelAsset != null)
        {
            runtimeModel = ModelLoader.Load(modelAsset);
            worker = new Worker(runtimeModel, BackendType.GPUCompute);
            inputTensor = new Tensor<float>(new TensorShape(1, 3, IMAGE_SIZE, IMAGE_SIZE));
            Debug.Log("AI dan Kamera Quest Siap!");
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
        if (worker == null || inputTensor == null) return;

        TextureTransform transform = new TextureTransform().SetDimensions(IMAGE_SIZE, IMAGE_SIZE).SetTensorLayout(TensorLayout.NCHW);
        TextureConverter.ToTensor(webcamTexture, inputTensor, transform);
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

            List<BoundingBox> smoothedBoxes = new List<BoundingBox>();
            smoothedBoxes.Add(previousSmoothedBox);
            DrawBoxes(smoothedBoxes);
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

        Vector2 uiSize = displayImage.rectTransform.rect.size;

        foreach (var box in boxesToDraw)
        {
            GameObject newBox = Instantiate(boxPrefab, displayImage.transform);
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