using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class UDPReceiver : MonoBehaviour
{
    public GameObject cube;
    public int port = 5052;

    UdpClient client;
    Thread thread;

    void Start()
    {
        client = new UdpClient(port);
        thread = new Thread(new ThreadStart(ReceiveData));
        thread.IsBackground = true;
        thread.Start();
    }

    void ReceiveData()
    {
        while (true)
        {
            IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
            byte[] data = client.Receive(ref anyIP);
            string json = Encoding.UTF8.GetString(data);

            Debug.Log("Received: " + json);

            // parsing simple (ambil object pertama)
            Detection[] detections = JsonHelper.FromJson<Detection>(json);

            if (detections.Length > 0)
            {
                Detection d = detections[0];

                // mapping ke posisi unity (simple)
                Vector3 pos = new Vector3(d.x / 100f, d.y / 100f, 5);
                cube.transform.position = pos;
            }
        }
    }

    [Serializable]
    public class Detection
    {
        public string label;
        public float x;
        public float y;
        public float w;
        public float h;
    }

    void OnApplicationQuit()
    {
        thread.Abort();
        client.Close();
    }
}

public class JsonHelper
{
    public static T[] FromJson<T>(string json)
    {
        string newJson = "{ \"array\": " + json + "}";
        Wrapper<T> wrapper = JsonUtility.FromJson<Wrapper<T>>(newJson);
        return wrapper.array;
    }

    [System.Serializable]
    private class Wrapper<T>
    {
        public T[] array;
    }
}