# CNN 2D — Diagnóstico do Redutor Planetário

Arquivos:
- `app.py` — aplicação Streamlit
- `modelo_redutor_audio.keras` — modelo treinado
- `requirements.txt` — dependências

## Execução local

```bash
pip install -r requirements.txt
streamlit run app.py
```

O app reproduz o pré-processamento do notebook:
Áudio -> mono/48 kHz -> 10 s -> Log-Mel Spectrogram -> CNN 2D.

Parâmetros:
- SR = 48000 Hz
- duração = 10 s
- N_FFT = 4096
- HOP_LENGTH = 1024
- N_MELS = 128
- FMAX = 24000 Hz

Classes:
1. REFERENCIA
2. DESALINHAMENTO
3. DESBALANCEAMENTO
4. SOBRECARGA
