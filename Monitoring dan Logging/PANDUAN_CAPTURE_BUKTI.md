# Panduan Capture Bukti Monitoring

Gunakan **PowerShell**, bukan Command Prompt. Jalankan setiap blok dari root
repository agregat; setiap blok dapat langsung dicopy ke terminal baru.

## 1. Bukti Serving

### Terminal 1: Jalankan exporter dan API

```powershell
$env:MODEL_URI = (Resolve-Path ".\Membangun_model\mlruns\922817180905645765\a101eca9146c4bd5b650b36df8667d06\artifacts\model").Path
& ".\Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto\.venv\Scripts\python.exe" ".\Monitoring dan Logging\3.prometheus_exporter.py" --host 127.0.0.1 --port 8088
```

Contoh setara untuk Unix shell:

```sh
export MODEL_URI="$(pwd)/Membangun_model/mlruns/922817180905645765/a101eca9146c4bd5b650b36df8667d06/artifacts/model"
python "Monitoring dan Logging/3.prometheus_exporter.py" --host 127.0.0.1 --port 8088
```

Biarkan terminal ini terbuka. Screenshot output saat model telah dimuat dan Uvicorn berjalan. Simpan sebagai:

```text
1.bukti_serving\prometheus_exporter.png
```

Jika muncul `Address already in use`, cari terminal lama yang menjalankan exporter pada port 8088 lalu tekan `Ctrl+C`. Jangan menghentikan Docker, Prometheus, atau Grafana.

### Terminal 2: Jalankan inference

```powershell
& ".\Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto\.venv\Scripts\python.exe" ".\Monitoring dan Logging\7.inference.py" --n 10
```

Screenshot output yang memuat `health_status: 200`, `http_status: 200`, `predictions`, dan `probabilities`. Simpan sebagai:

```text
1.bukti_serving\inference.png
```

## 2. Jalankan Prometheus dan Grafana

Terminal 1 exporter harus tetap hidup. Di terminal baru, jalankan:

```powershell
docker compose -f ".\Monitoring dan Logging\docker-compose.yml" up -d
docker compose -f ".\Monitoring dan Logging\docker-compose.yml" ps
```

Bila inference belum dijalankan setelah Compose aktif, ulangi perintah inference pada bagian 1 untuk mengisi metrik.

## 3. Bukti Monitoring Prometheus

Buka target scrape untuk screenshot:

```powershell
Start-Process "http://127.0.0.1:9090/targets"
Start-Process "http://127.0.0.1:9090/graph"
```

Di halaman `Graph`, jalankan satu query per screenshot berikut:

```promql
model_up
```

```promql
prediction_requests_total
```

```promql
prediction_latency_seconds_count
```

```promql
process_memory_bytes
```

```promql
exporter_uptime_seconds
```

Simpan screenshot di folder `4.bukti monitoring Prometheus` dengan nama:

```text
monitoring_model_up.png
monitoring_prediction_requests_total.png
monitoring_prediction_latency_seconds.png
monitoring_process_memory_bytes.png
monitoring_exporter_uptime_seconds.png
```

## 4. Bukti Monitoring Grafana

Buka Grafana:

```powershell
Start-Process "http://127.0.0.1:3000"
```

Login dengan:

```text
username: admin
password: admin
```

Pilih **Dashboards**, lalu buka dashboard:

```text
Bank Marketing ML Monitoring - muhammad_yusuf_tdrv4
```

Pastikan terlihat panel metric dan judul dashboard. Simpan sebagai:

```text
5.bukti monitoring Grafana\dashboard_12_metrics.png
```

## 5. Bukti Alerting

Tiga rule Prometheus sudah disediakan:

```text
ModelServiceDown
HighPredictionErrorRate
HighPredictionLatencyP95
```

Buka halaman rule Prometheus:

```powershell
Start-Process "http://127.0.0.1:9090/rules"
Start-Process "http://127.0.0.1:9090/alerts"
```

Untuk bukti dari Grafana, buka **Alerting** lalu pilih data source-managed alert rules dari Prometheus. Capture rule yang terlihat dan simpan sebagai:

```text
6.bukti alerting Grafana\rules_model_service_down.png
6.bukti alerting Grafana\rules_high_prediction_error_rate.png
6.bukti alerting Grafana\rules_high_prediction_latency_p95.png
```

Untuk memicu `ModelServiceDown`, hentikan Terminal 1 exporter dengan `Ctrl+C`, tunggu lebih dari satu menit, lalu refresh halaman `/alerts` atau halaman Alerting Grafana. Screenshot state firing sebagai:

```text
6.bukti alerting Grafana\notification_model_service_down.png
```

Jalankan kembali exporter setelah screenshot:

```powershell
$env:MODEL_URI = (Resolve-Path ".\Membangun_model\mlruns\922817180905645765\a101eca9146c4bd5b650b36df8667d06\artifacts\model").Path
& ".\Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto\.venv\Scripts\python.exe" ".\Monitoring dan Logging\3.prometheus_exporter.py" --host 127.0.0.1 --port 8088
```

`HighPredictionErrorRate` dan `HighPredictionLatencyP95` memerlukan traffic error/latensi nyata untuk firing. Jangan membuat screenshot notification jika rule belum benar-benar firing atau belum ada contact point Grafana.
