import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf


# ============================================================
# CONFIGURAÇÃO — igual ao notebook CNN_SONORO_V2
# ============================================================

TARGET_SR = 48000
TEMPO_MAX = 10

N_FFT = 4096
HOP_LENGTH = 1024
N_MELS = 128
FMAX = TARGET_SR // 2

CLASS_NAMES = [
    "REFERENCIA",
    "DESALINHAMENTO",
    "DESBALANCEAMENTO",
    "SOBRECARGA",
]

MODEL_PATH = os.environ.get("MODEL_PATH", "modelo_redutor_audio.keras")


# ============================================================
# CARREGAMENTO DO MODELO
# ============================================================

@st.cache_resource
def carregar_modelo():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modelo não encontrado: {MODEL_PATH}. "
            "Coloque o arquivo .keras na mesma pasta do app.py "
            "ou altere MODEL_PATH."
        )

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


# ============================================================
# PRÉ-PROCESSAMENTO — igual ao notebook
# ============================================================

def carregar_audio_fixo(caminho, sr=TARGET_SR, duracao=TEMPO_MAX):
    audio, _ = librosa.load(
        caminho,
        sr=sr,
        mono=True
    )

    amostras = int(sr * duracao)

    if len(audio) < amostras:
        audio = np.pad(
            audio,
            (0, amostras - len(audio)),
            mode="constant"
        )
    else:
        audio = audio[:amostras]

    return audio


def gerar_log_mel(audio):
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=FMAX,
        power=2.0
    )

    log_mel_spec = librosa.power_to_db(
        mel_spec,
        ref=np.max
    )

    return log_mel_spec


def preparar_entrada(log_mel_spec):
    # O modelo espera:
    # (batch, n_mels, n_frames, canais)
    X = np.asarray(log_mel_spec, dtype=np.float32)
    X = X[np.newaxis, ..., np.newaxis]
    return X


# ============================================================
# INTERFACE
# ============================================================

st.set_page_config(
    page_title="Diagnóstico do Redutor Planetário",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Diagnóstico de Redutor Planetário")
st.write(
    "Classificação de uma gravação de áudio por meio de uma "
    "CNN 2D utilizando o espectrograma Mel em escala logarítmica."
)

st.info(
    "Fluxo utilizado: Áudio → Log-Mel Spectrogram → CNN 2D → Classificação"
)

arquivo = st.file_uploader(
    "Selecione um arquivo de áudio",
    type=["wav", "mp3", "m4a"],
    help="O áudio será convertido para mono, reamostrado para 48 kHz "
         "e ajustado para 10 segundos, conforme o processamento do notebook."
)

if arquivo is not None:

    st.audio(arquivo)

    # Salva temporariamente o arquivo enviado
    suffix = os.path.splitext(arquivo.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(arquivo.getbuffer())
        caminho_audio = temp.name

    try:
        with st.spinner("Processando o áudio..."):

            # 1. Carrega e ajusta o áudio
            audio = carregar_audio_fixo(caminho_audio)

            # 2. Gera Log-Mel Spectrogram
            log_mel_spec = gerar_log_mel(audio)

            # 3. Prepara entrada da CNN
            X = preparar_entrada(log_mel_spec)

            # 4. Carrega modelo
            model = carregar_modelo()

            # 5. Predição
            probabilidades = model.predict(X, verbose=0)[0]

            indice_predito = int(np.argmax(probabilidades))
            classe_predita = CLASS_NAMES[indice_predito]
            confianca = float(probabilidades[indice_predito])

        st.success("Classificação concluída!")

        # ----------------------------------------------------
        # RESULTADO
        # ----------------------------------------------------

        st.subheader("Resultado")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Condição identificada",
                classe_predita
            )

        with col2:
            st.metric(
                "Confiança",
                f"{confianca * 100:.2f}%"
            )

        # ----------------------------------------------------
        # PROBABILIDADES
        # ----------------------------------------------------

        st.subheader("Probabilidade por classe")

        df_prob = pd.DataFrame({
            "Condição": CLASS_NAMES,
            "Probabilidade": probabilidades * 100
        })

        df_prob["Probabilidade"] = df_prob["Probabilidade"].round(2)

        st.dataframe(
            df_prob,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            df_prob.set_index("Condição")["Probabilidade"]
        )

        # ----------------------------------------------------
        # LOG-MEL SPECTROGRAM
        # ----------------------------------------------------

        st.subheader("Espectrograma Mel em escala logarítmica")

        fig, ax = plt.subplots(figsize=(12, 5))

        img = librosa.display.specshow(
            log_mel_spec,
            sr=TARGET_SR,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            fmax=FMAX,
            ax=ax
        )

        ax.set_title("Log-Mel Spectrogram")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Frequência Mel")

        fig.colorbar(img, ax=ax, format="%+2.0f dB")

        st.pyplot(fig)
        plt.close(fig)

        # ----------------------------------------------------
        # INFORMAÇÕES TÉCNICAS
        # ----------------------------------------------------

        with st.expander("Informações do processamento"):
            st.write(f"**Taxa de amostragem:** {TARGET_SR} Hz")
            st.write(f"**Duração utilizada:** {TEMPO_MAX} s")
            st.write(f"**N_FFT:** {N_FFT}")
            st.write(f"**Hop Length:** {HOP_LENGTH}")
            st.write(f"**Número de bandas Mel:** {N_MELS}")
            st.write(f"**Frequência máxima:** {FMAX} Hz")
            st.write(f"**Formato do Log-Mel:** {log_mel_spec.shape}")
            st.write(f"**Entrada da CNN:** {X.shape}")

    except Exception as e:
        st.error(f"Erro ao processar o áudio: {e}")

    finally:
        if os.path.exists(caminho_audio):
            os.remove(caminho_audio)

else:
    st.warning("Envie um arquivo de áudio para iniciar a classificação.")
