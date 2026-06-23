using UnityEngine;


public class ObjectDetector : MonoBehaviour
{
    [Header("Sentis")]
    public Unity.InferenceEngine.ModelAsset modelAsset;

    private Unity.InferenceEngine.Model runtimeModel;
    private Unity.InferenceEngine.Worker worker;

    void Start()
    {
        if (modelAsset == null)
        {
            Debug.LogError("ModelAsset belum di-assign!");
            return;
        }

        // Load model dari asset
        runtimeModel = Unity.InferenceEngine.ModelLoader.Load(modelAsset);

        // Buat worker (CPU dulu, paling aman)
        worker = new Unity.InferenceEngine.Worker(runtimeModel, Unity.InferenceEngine.BackendType.CPU);

        Debug.Log("Sentis Worker berhasil dibuat!");
    }

    void OnDestroy()
    {
        // WAJIB dispose (penting di Quest)
        worker?.Dispose();
    }
}
