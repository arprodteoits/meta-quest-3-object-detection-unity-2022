using UnityEngine;

public class BoundingBoxRenderer : MonoBehaviour
{
    public RectTransform boxPrefab;
    public Canvas canvas;

    public void DrawBox(Rect rect)
    {
        RectTransform box = Instantiate(boxPrefab, canvas.transform);
        box.anchoredPosition = rect.position;
        box.sizeDelta = rect.size;
    }
}
