using UnityEngine;
using Unity.Sentis;

float inferenceInterval = 0.2f; 
float lastInferenceTime = 0f;

public class YOLODetector : MonoBehaviour
{
    public ModelAsset modelAsset;

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
    }

    void Update()
    {
        if (webcam.width <= 16) return;

        // Convert webcam texture → tensor
        var input = TextureConverter.ToTensor(webcam, 640, 640, 3);

        // Run inference
        worker.Schedule(input);

        // Get output
        var output = worker.PeekOutput();

        Debug.Log("YOLO inference running");
        Debug.Log("Output shape: " + output.shape);
    }

    void OnDestroy()
    {
        webcam.Stop();
        worker.Dispose();
    }
}