using UnityEngine;

public class CameraCapture : MonoBehaviour
{
    public Camera arCamera;
    public RenderTexture renderTexture;

    void Start()
    {
        renderTexture = new RenderTexture(640, 640, 24);
        arCamera.targetTexture = renderTexture;
    }

    public Texture GetTexture()
    {
        return renderTexture;
    }
}
