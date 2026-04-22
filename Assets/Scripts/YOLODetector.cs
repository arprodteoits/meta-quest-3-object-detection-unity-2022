using UnityEngine;
using Unity.Sentis;



public class YOLODetector : MonoBehaviour
{
    public ModelAsset modelAsset;

    float inferenceInterval = 0.2f; 
    float lastInferenceTime = 0f;

    private WebCamTexture webcam;
    private Worker worker;
    private Model runtimeModel;

    void Start()
    {
        // Start webcam
        webcam = new WebCamTexture();
        webcam.Play();

        // Load ONNX model
        runtimeModel = ModelLoader.Load(modelAsset);

        // Create worker
        worker = new Worker(runtimeModel, BackendType.GPUCompute);

        Debug.Log("YOLO started");
    }

void Update()
{
    if (webcam.width <= 16) return;

    if (Time.time - lastInferenceTime < inferenceInterval)
        return;

    lastInferenceTime = Time.time;

    var input = TextureConverter.ToTensor(webcam, 640, 640, 3);

    worker.Schedule(input);

    var output = worker.PeekOutput();

    Debug.Log("YOLO inference running");
}

    void OnDestroy()
    {
        webcam.Stop();
        worker.Dispose();
    }
}