using UnityEngine;

public class SpawnerInteraksi : MonoBehaviour
{
    // Masukkan objek baru (prefab) yang ingin dimunculkan dari Inspector
    public GameObject objekBaruPrefab; 

    // Fungsi ini otomatis berjalan HANYA SAAT ada objek lain menyentuh Collider
    void OnTriggerEnter(Collider objekYangMenabrak)
    {
        // Cek apakah yang menabrak adalah instrumen yang benar (menggunakan Tag)
        if (objekYangMenabrak.gameObject.CompareTag("InstrumenB"))
        {
            Debug.Log("Objek bersentuhan! Memunculkan objek baru...");
            
            // Logika untuk memunculkan objek baru (Instantiate) di posisi sentuhan
            Instantiate(objekBaruPrefab, transform.position, Quaternion.identity);
        }
    }
}