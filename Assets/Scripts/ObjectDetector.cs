using UnityEngine;
using Unity.Sentis;

public class ObjectDetector : MonoBehaviour
{
    [Header("Sentis")]
    public ModelAsset modelAsset;

    private Model runtimeModel;
    private Worker worker;

    void Start()
    {
        if (modelAsset == null)
        {
            Debug.LogError("ModelAsset belum di-assign!");
            return;
        }

        // Load model dari asset
        runtimeModel = ModelLoader.Load(modelAsset);

        // Buat worker (CPU dulu, paling aman)
        worker = new Worker(runtimeModel, BackendType.CPU);

        Debug.Log("Sentis Worker berhasil dibuat!");
    }

    void OnDestroy()
    {
        // WAJIB dispose (penting di Quest)
        worker?.Dispose();
    }
}
