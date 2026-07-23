using UnityEngine;


public class ObjectDetector : MonoBehaviour
{
    [Header("Sentis")]
    public Unity.Sentis.ModelAsset modelAsset;

    private Unity.Sentis.Model runtimeModel;
    private Unity.Sentis.Worker worker;

    void Start()
    {
        if (modelAsset == null)
        {
            Debug.LogError("ModelAsset belum di-assign!");
            return;
        }

        // Load model dari asset
        runtimeModel = Unity.Sentis.ModelLoader.Load(modelAsset);

        // Buat worker (CPU dulu, paling aman)
        worker = new Unity.Sentis.Worker(runtimeModel, Unity.Sentis.BackendType.CPU);

        Debug.Log("Sentis Worker berhasil dibuat!");
    }

    void OnDestroy()
    {
        // WAJIB dispose (penting di Quest)
        worker?.Dispose();
    }
}
