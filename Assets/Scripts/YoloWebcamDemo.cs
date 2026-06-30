using System.Collections;
using System.Collections.Generic;
using UnityEngine;
       // ← MUST exist
using UnityEngine.UI;
using TMPro;
using UnityEngine.Android;

public class YoloWebcamDemo : MonoBehaviour
{
    [Header("Sentis AI")]
    public Unity.InferenceEngine.ModelAsset modelAsset;
    private Unity.InferenceEngine.Model runtimeModel;
    private Unity.InferenceEngine.Worker worker;
    private Unity.InferenceEngine.Tensor<float> inputTensor;

    private string[] classNames = { "bottle", "eraser", "pen" };

    [Header("Kamera & UI")]
    public RawImage displayImage;
    public GameObject boxPrefab;
    public ObjectInteractionManager interactionManager;

    // ✅ GANTI: Bukan WebCamTexture lagi, tapi RenderTexture
    private RenderTexture cameraRenderTexture;
    private Texture2D readbackTexture;
    private bool cameraReady = false;

    [Range(0.0f, 1.0f)]
    public float confidenceThreshold = 0.35f;

    private List<GameObject> activeBoxes = new List<GameObject>();
    private const int NUM_PROPOSALS = 8400;
    private const int IMAGE_SIZE = 640;

    public struct BoundingBox {
        public float cx, cy, w, h, conf;
        public string label;
    }

void Start()
{
    Debug.Log("[YOLO-DEBUG] Start() called - Sentis 2.5.0");

    if (modelAsset == null) {
        Debug.LogError("[YOLO-DEBUG] modelAsset is NULL!");
        return;
    }

    // ✅ Sentis 2.5.0 correct API
    runtimeModel = Unity.InferenceEngine.ModelLoader.Load(modelAsset);
    worker = new Unity.InferenceEngine.Worker(runtimeModel, Unity.InferenceEngine.BackendType.CPU);
    inputTensor = new Unity.InferenceEngine.Tensor<float>(new Unity.InferenceEngine.TensorShape(1, 3, IMAGE_SIZE, IMAGE_SIZE));
    
    Debug.Log("[YOLO-DEBUG] Model loaded OK - Sentis 2.5.0");

    // Camera permission
    if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
    {
        Permission.RequestUserPermission(Permission.Camera);
        StartCoroutine(WaitForPermissionThenInit());
    }
    else
    {
        InitCamera();
    }
}

IEnumerator WaitForPermissionThenInit()
{
    Debug.Log("[YOLO-DEBUG] Waiting for permission...");
    float timeout = 10f;
    float elapsed = 0f;
    
    while (!Permission.HasUserAuthorizedPermission(Permission.Camera))
    {
        elapsed += Time.deltaTime;
        if (elapsed > timeout) {
            Debug.LogError("[YOLO-DEBUG] Permission timeout! User may have denied camera.");
            yield break;
        }
        yield return null;
    }
    
    Debug.Log("[YOLO-DEBUG] Permission granted after " + elapsed + "s");
    InitCamera();
}

   void InitCamera()
{
    Debug.Log("[YOLO-DEBUG] InitCamera() called");
    
    WebCamDevice[] devices = WebCamTexture.devices;
    Debug.Log("[YOLO-DEBUG] Camera devices found: " + devices.Length);

    for (int i = 0; i < devices.Length; i++)
        Debug.Log($"[YOLO-DEBUG] Device[{i}]: {devices[i].name}");

    if (devices.Length == 0)
    {
        Debug.LogError("[YOLO-DEBUG] NO CAMERAS FOUND! Permission may be denied.");
        return;
    }

    string camName = devices[0].name;
    Debug.Log("[YOLO-DEBUG] Using camera: " + camName);

    activeWebcam = new WebCamTexture(camName, 640, 480, 30);
    cameraRenderTexture = new RenderTexture(IMAGE_SIZE, IMAGE_SIZE, 0, RenderTextureFormat.ARGB32);

    if (displayImage != null)
        displayImage.texture = activeWebcam;

    activeWebcam.Play();
    Debug.Log("[YOLO-DEBUG] webcam.Play() called");
    StartCoroutine(WaitForCameraStart());
}

  IEnumerator WaitForCameraStart()
{
    Debug.Log("[YOLO-DEBUG] Waiting for camera to start...");
    float timeout = 10f;
    float elapsed = 0f;
    
    while (activeWebcam.width <= 16)
    {
        elapsed += Time.deltaTime;
        if (elapsed > timeout) {
            Debug.LogError("[YOLO-DEBUG] Camera start timeout! width=" + activeWebcam.width);
            yield break;
        }
        yield return null;
    }
    
    Debug.Log($"[YOLO-DEBUG] Camera started! Size: {activeWebcam.width}x{activeWebcam.height}");
    cameraReady = true;
}

    private WebCamTexture activeWebcam;

    void Update()
    {
        if (cameraReady && activeWebcam != null && activeWebcam.didUpdateThisFrame)
        {
            ExecuteInference();
        }
    }

    void ExecuteInference()
    {
        // ✅ Blit webcam ke RenderTexture 640x640 (resize otomatis)
        Graphics.Blit(activeWebcam, cameraRenderTexture);

        // ✅ Konversi RenderTexture ke Tensor untuk YOLO
        Unity.InferenceEngine.TextureTransform transform = new Unity.InferenceEngine.TextureTransform()
            .SetDimensions(IMAGE_SIZE, IMAGE_SIZE)
            .SetTensorLayout(Unity.InferenceEngine.TensorLayout.NCHW);

        Unity.InferenceEngine.TextureConverter.ToTensor(cameraRenderTexture, inputTensor, transform);
        worker.Schedule(inputTensor);

        Unity.InferenceEngine.Tensor<float> outputTensor = worker.PeekOutput() as Unity.InferenceEngine.Tensor<float>;
        if (outputTensor != null)
        {
            float[] data = outputTensor.DownloadToArray();
            ParseYOLOOutput(data);
        }
    }

    void ParseYOLOOutput(float[] data)
    {
        List<BoundingBox> boxes = new List<BoundingBox>();
        int numClasses = classNames.Length;

        for (int i = 0; i < NUM_PROPOSALS; i++)
        {
            float maxScore = 0;
            int bestClassId = 0;

            for (int c = 0; c < numClasses; c++)
            {
                float score = data[(4 + c) * NUM_PROPOSALS + i];
                if (score > maxScore) { maxScore = score; bestClassId = c; }
            }

            if (maxScore > confidenceThreshold)
            {
                boxes.Add(new BoundingBox {
                    cx = data[0 * NUM_PROPOSALS + i],
                    cy = data[1 * NUM_PROPOSALS + i],
                    w  = data[2 * NUM_PROPOSALS + i],
                    h  = data[3 * NUM_PROPOSALS + i],
                    conf = maxScore,
                    label = classNames[bestClassId]
                });
            }
        }

        // NMS
        boxes.Sort((a, b) => b.conf.CompareTo(a.conf));
        List<BoundingBox> finalBoxes = new List<BoundingBox>();
        foreach (var box in boxes)
        {
            bool overlap = false;
            foreach (var fb in finalBoxes)
            {
                if (CalculateIoU(box, fb) > 0.45f) { overlap = true; break; }
            }
            if (!overlap) finalBoxes.Add(box);
        }

        // Kirim ke InteractionManager
        if (interactionManager != null)
        {
            Vector2 uiSize = displayImage.rectTransform.rect.size;
            List<ObjectInteractionManager.DetectedObject> interactionList =
                new List<ObjectInteractionManager.DetectedObject>();

            foreach (var b in finalBoxes)
            {
                float xNorm = b.cx / IMAGE_SIZE;
                float yNorm = b.cy / IMAGE_SIZE;
                float uiX = (xNorm * uiSize.x) - (uiSize.x / 2f);
                float uiY = (uiSize.y / 2f) - (yNorm * uiSize.y);

                interactionList.Add(new ObjectInteractionManager.DetectedObject {
                    label = b.label,
                    centroid = new Vector2(uiX, uiY)
                });
            }
            interactionManager.UpdateDetections(interactionList);
        }

        DrawBoxes(finalBoxes);
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
            rt.anchorMin = rt.anchorMax = rt.pivot = new Vector2(0.5f, 0.5f);

            float xNorm = box.cx / IMAGE_SIZE;
            float yNorm = box.cy / IMAGE_SIZE;
            float uiX = (xNorm * uiSize.x) - (uiSize.x / 2f);
            float uiY = (uiSize.y / 2f) - (yNorm * uiSize.y);

            rt.anchoredPosition = new Vector2(uiX, uiY);
            rt.sizeDelta = new Vector2(
                (box.w / IMAGE_SIZE) * uiSize.x,
                (box.h / IMAGE_SIZE) * uiSize.y);

            var textComponent = newBox.GetComponentInChildren<TextMeshProUGUI>();
            if (textComponent != null)
                textComponent.text = $"{box.label} {(box.conf * 100):0}%";

            activeBoxes.Add(newBox);
        }
    }

    float CalculateIoU(BoundingBox a, BoundingBox b)
    {
        float xA = Mathf.Max(a.cx - a.w / 2, b.cx - b.w / 2);
        float yA = Mathf.Max(a.cy - a.h / 2, b.cy - b.h / 2);
        float xB = Mathf.Min(a.cx + a.w / 2, b.cx + b.w / 2);
        float yB = Mathf.Min(a.cy + a.h / 2, b.cy + b.h / 2);
        float interArea = Mathf.Max(0, xB - xA) * Mathf.Max(0, yB - yA);
        return interArea / (a.w * a.h + b.w * b.h - interArea);
    }

    private void OnDisable()
    {
        worker?.Dispose();
        inputTensor?.Dispose();
        if (activeWebcam != null) activeWebcam.Stop();
        if (cameraRenderTexture != null) cameraRenderTexture.Release();
    }
}