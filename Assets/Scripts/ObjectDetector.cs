using UnityEngine;
using Unity.Sentis;

public class ObjectDetector : MonoBehaviour
{
    public CameraCapture cameraCapture;
    public ModelAsset modelAsset;

    private Model runtimeModel;
    private Worker worker;

    const int INPUT_SIZE = 640;

    void Start()
    {
        runtimeModel = ModelLoader.Load(modelAsset);

        worker = new Worker(
            runtimeModel,
            BackendType.GPUCompute
        );
    }

    void Update()
    {
        RunDetection();
    }

    void RunDetection()
    {
        Texture tex = cameraCapture.GetTexture();

        // ✅ API LAMA — MASIH VALID
        Tensor<float> inputTensor =
            TextureConverter.ToTensor(tex, INPUT_SIZE, INPUT_SIZE, 3);

        worker.Schedule(inputTensor);

        Tensor output = worker.PeekOutput();
        Debug.Log("Inference OK: " + output.shape);

        inputTensor.Dispose();
    }

    void OnDestroy()
    {
        worker.Dispose();
    }
}
