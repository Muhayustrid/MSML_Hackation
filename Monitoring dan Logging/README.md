# Monitoring dan Logging

Bank Marketing serving and monitoring for `muhammad_yusuf_tdrv4`.

## Deterministic Local Serving

Run from the aggregate repository root. The selected artifact is the final
manual tuning run `a101eca9146c4bd5b650b36df8667d06`.

PowerShell:

```powershell
$env:MODEL_URI = (Resolve-Path ".\Membangun_model\mlruns\922817180905645765\a101eca9146c4bd5b650b36df8667d06\artifacts\model").Path
& ".\Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto\.venv\Scripts\python.exe" ".\Monitoring dan Logging\3.prometheus_exporter.py" --host 127.0.0.1 --port 8088
```

Unix shell:

```sh
export MODEL_URI="$(pwd)/Membangun_model/mlruns/922817180905645765/a101eca9146c4bd5b650b36df8667d06/artifacts/model"
python "Monitoring dan Logging/3.prometheus_exporter.py" --host 127.0.0.1 --port 8088
```

## Local URLs

- API documentation: `http://127.0.0.1:8088/docs`
- Health: `http://127.0.0.1:8088/health`
- Metrics: `http://127.0.0.1:8088/metrics`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`

## Draft Evidence

The numbered evidence folders follow the accepted reference structure. GUI screenshots and public-repository evidence remain manual submission gates; they are not fabricated by this package.
