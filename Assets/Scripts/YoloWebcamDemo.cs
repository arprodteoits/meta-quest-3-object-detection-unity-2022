using UnityEngine;
using Unity.Sentis;
using UnityEngine.UI;

public class YoloWebcamDemo : MonoBehaviour
{
    public ModelAsset modelAsset;
    public RawImage displayImage; // Untuk nampilin video webcam
    
    private Worker worker;
    private Tensor<float> inputTensor;
    private WebCamTexture webcamTexture;
    private Model runtimeModel;

    void Start()
    {
        // 1. Setup Camera
        webcamTexture = new WebCamTexture();
        webcamTexture.Play();
        displayImage.texture = webcamTexture;

        // 2. Setup Sentis
        runtimeModel = ModelLoader.Load(modelAsset);
        // Kita gunakan GPU (Compute) agar cepat di PC
        worker = new Worker(runtimeModel, BackendType.GPUCompute);
    }

    void Update()
    {
        if (webcamTexture.didUpdateThisFrame)
        {
            ExecuteInference();
        }
    }

    void ExecuteInference()
    {
        // 3. Konversi Texture ke Tensor
        // YOLOv8 biasanya butuh input 640x640 atau 320x320
        using Tensor inputTensor = TextureConverter.ToTensor(webcamTexture, 640, 640, 3);
        
        // 4. Jalankan AI
        worker.Schedule(inputTensor);

        // 5. Ambil Output
        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
        
        // Di sini kamu perlu melakukan Post-Processing (NMS & Bounding Box)
        // Untuk demo awal, kita cek apakah ada data keluar
        Debug.Log("Output tensor shape: " + outputTensor.shape);
    }

    private void OnDisable()
    {
        worker?.Dispose();
    }
}