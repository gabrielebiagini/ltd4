import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import requests
import os
import io
from lime import lime_image
from skimage.segmentation import mark_boundaries
from serpapi import SerpApiClient
import gspread
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="Analisi Funghi con XAI Avanzato",
    page_icon="🍄",
    layout="wide"
)

# --- DATI E MODELLO ---
try:
    with open('class_labels.txt', 'r') as f:
        SPECIES_LIST = [line.strip() for line in f]
except FileNotFoundError:
    st.error("Errore critico: il file 'class_labels.txt' non è stato trovato.")
    st.stop()

# Incolla qui il tuo dizionario FUNGI_INFO completo
FUNGI_INFO = {
    "Agaricus bisporus": {"nome_italiano": "Prataiolo coltivato", "commestibile": "Commestibile"},
    "Agaricus subrufescens": {"nome_italiano": "Prataiolo mandorlato", "commestibile": "Commestibile"},
    "Amanita bisporigera": {"nome_italiano": "Amanita bisporigera", "commestibile": "Velenoso"},
    "Amanita muscaria": {"nome_italiano": "Amanita muscaria", "commestibile": "Velenoso"},
    "Amanita ocreata": {"nome_italiano": "Amanita ocreata", "commestibile": "Velenoso"},
    "Amanita phalloides": {"nome_italiano": "Amanita falloide", "commestibile": "Mortale"},
    "Amanita smithiana": {"nome_italiano": "Amanita smithiana", "commestibile": "Velenoso"},
    "Amanita verna": {"nome_italiano": "Amanita verna", "commestibile": "Mortale"},
    "Amanita virosa": {"nome_italiano": "Amanita virosa", "commestibile": "Mortale"},
    "Auricularia auricula-judae": {"nome_italiano": "Orecchio di Giuda", "commestibile": "Commestibile"},
    "Boletus edulis": {"nome_italiano": "Porcino", "commestibile": "Commestibile"},
    "Cantharellus cibarius": {"nome_italiano": "Gallinaccio", "commestibile": "Commestibile"},
    "Clitocybe dealbata": {"nome_italiano": "Clitocybe dealbata", "commestibile": "Velenoso"},
    "Conocybe filaris": {"nome_italiano": "Conocybe filaris", "commestibile": "Velenoso"},
    "Coprinus comatus": {"nome_italiano": "Coprino chiomato", "commestibile": "Commestibile (con cautela)"},
    "Cordyceps sinensis": {"nome_italiano": "Cordyceps", "commestibile": "Utilizzato in medicina tradizionale"},
    "Cortinarius rubellus": {"nome_italiano": "Cortinarius rubellus", "commestibile": "Mortale"},
    "Entoloma sinuatum": {"nome_italiano": "Entoloma sinuatum", "commestibile": "Velenoso"},
    "Flammulina velutipes": {"nome_italiano": "Fammulina", "commestibile": "Commestibile"},
    "Galerina marginata": {"nome_italiano": "Galerina marginata", "commestibile": "Mortale"},
    "Ganoderma lucidum": {"nome_italiano": "Reishi", "commestibile": "Utilizzato in medicina tradizionale"},
    "Grifola frondosa": {"nome_italiano": "Maitake", "commestibile": "Commestibile"},
    "Gyromitra esculenta": {"nome_italiano": "Gyromitra esculenta", "commestibile": "Commestibile (con preparazione speciale)"},
    "Hericium erinaceus": {"nome_italiano": "Criniera di leone", "commestibile": "Commestibile"},
    "Hydnum repandum": {"nome_italiano": "Steccherino dorato", "commestibile": "Commestibile"},
    "Hypholoma fasciculare": {"nome_italiano": "Hypholoma fasciculare", "commestibile": "Velenoso"},
    "Inocybe erubescens": {"nome_italiano": "Inocybe erubescens", "commestibile": "Velenoso"},
    "Lentinula edodes": {"nome_italiano": "Shiitake", "commestibile": "Commestibile"},
    "Lepiota brunneoincarnata": {"nome_italiano": "Lepiota brunneoincarnata", "commestibile": "Mortale"},
    "Macrolepiota procera": {"nome_italiano": "Mazza di tamburo", "commestibile": "Commestibile"},
    "Morchella esculenta": {"nome_italiano": "Spugnola comune", "commestibile": "Commestibile (con preparazione speciale)"},
    "Omphalotus olearius": {"nome_italiano": "Omphalotus olearius", "commestibile": "Velenoso"},
    "Paxillus involutus": {"nome_italiano": "Paxillus involutus", "commestibile": "Velenoso"},
    "Pholiota nameko": {"nome_italiano": "Nameko", "commestibile": "Commestibile"},
    "Pleurotus citrinopileatus": {"nome_italiano": "Pleurotus citrinopileatus", "commestibile": "Commestibile"},
    "Pleurotus eryngii": {"nome_italiano": "Cardoncello", "commestibile": "Commestibile"},
    "Pleurotus ostreatus": {"nome_italiano": "Orecchione", "commestibile": "Commestibile"},
    "Psilocybe semilanceata": {"nome_italiano": "Psilocybe semilanceata", "commestibile": "Allucinogeno"},
    "Rhodophyllus rhodopolius": {"nome_italiano": "Rhodophyllus rhodopolius", "commestibile": "Velenoso"},
    "Russula emetica": {"nome_italiano": "Colombina rossa", "commestibile": "Velenoso"},
    "Russula virescens": {"nome_italiano": "Colombina verde", "commestibile": "Commestibile"},
    "Scleroderma citrinum": {"nome_italiano": "Falso tartufo", "commestibile": "Velenoso"},
    "Suillus luteus": {"nome_italiano": "Pinarolo", "commestibile": "Commestibile"},
    "Tremella fuciformis": {"nome_italiano": "Tremella fuciformis", "commestibile": "Commestibile"},
    "Tricholoma matsutake": {"nome_italiano": "Matsutake", "commestibile": "Commestibile"},
    "Truffles": {"nome_italiano": "Tartufo", "commestibile": "Commestibile"},
    "Tuber melanosporum": {"nome_italiano": "Tartufo nero pregiato", "commestibile": "Commestibile"}
}

# NUOVO: Fattori di calibrazione illustrativi
CALIBRATION_FACTORS = {
    "Amanita phalloides": 0.92, # Tende a essere troppo sicuro
    "Boletus edulis": 1.08,    # Tende a essere troppo cauto
    "default": 1.0
}

# --- NUOVE FUNZIONI XAI ---

def generate_natural_language_explanation(predicted_species, confidence, top_3_predictions):
    """Genera spiegazione in linguaggio naturale."""
    info = FUNGI_INFO.get(predicted_species, {"nome_italiano": "N/A", "commestibile": "N/A"})
    confidence_text = "molto sicuro" if confidence > 90 else "abbastanza sicuro" if confidence > 70 else "moderatamente sicuro" if confidence > 50 else "incerto"
    
    explanation = f"Il modello è **{confidence_text}** ({confidence:.1f}%) che questo sia un **{info['nome_italiano']}** (*{predicted_species}*). "
    explanation += f"La commestibilità riportata è: **{info['commestibile']}**."
    
    explanation += "\n\n**Alternative considerate:**"
    # Mostra la 2a e 3a alternativa
    for species, conf in top_3_predictions[1:]:
        species_info = FUNGI_INFO.get(species, {"nome_italiano": "N/A"})
        explanation += f"\n- L'alternativa più probabile era *{species_info['nome_italiano']}* con il {conf:.1f}%."
        
    if info['commestibile'] in ['Velenoso', 'Mortale']:
        explanation += "\n\n🚨 **ATTENZIONE: Non consumare mai funghi senza il parere di un esperto.**"
        
    return explanation

def generate_contrastive_explanation(all_confidences, species_list):
    """Genera spiegazione contrastiva (perché A e non B)."""
    sorted_preds = sorted(enumerate(all_confidences), key=lambda x: x[1], reverse=True)
    
    first_class_idx, first_class_conf = sorted_preds[0]
    second_class_idx, second_class_conf = sorted_preds[1]
    
    first_info = FUNGI_INFO.get(species_list[first_class_idx], {"nome_italiano": "N/A"})
    second_info = FUNGI_INFO.get(species_list[second_class_idx], {"nome_italiano": "N/A"})
    
    explanation = f"""
    #### Perché *{first_info['nome_italiano']}* e non *{second_info['nome_italiano']}*?

    Il modello ha preferito **{first_info['nome_italiano']}** ({first_class_conf:.1f}%) rispetto alla seconda opzione più probabile, **{second_info['nome_italiano']}** ({second_class_conf:.1f}%), con una differenza di **{first_class_conf - second_class_conf:.1f}** punti percentuali.

    Questo suggerisce che le caratteristiche visive dell'immagine (come la forma del cappello o la colorazione) corrispondono in modo più significativo ai pattern che il modello ha imparato ad associare alla specie *{first_info['nome_italiano']}*.

    **Simulazione controfattuale (testuale):**
    - Se l'immagine avesse mostrato un **cappello di forma leggermente diversa**, la probabilità per *{second_info['nome_italiano']}* sarebbe potuta aumentare.
    """
    return explanation
@st.cache_data
def monte_carlo_uncertainty(_model, image_array, n_samples=20):
    predictions = []
    for _ in range(n_samples):
        noise = np.random.normal(0, 0.01, image_array.shape)
        pred = _model.predict(np.clip(image_array + noise, 0, 1), verbose=0)
        predictions.append(pred[0])
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    return mean_pred, std_pred

def create_uncertainty_visualization(mean_pred, std_pred, species_list):
    top_5_indices = np.argsort(mean_pred)[-5:][::-1]
    species_names = [FUNGI_INFO.get(species_list[i], {"nome_italiano": species_list[i]})["nome_italiano"] for i in top_5_indices]
    means = [mean_pred[i] * 100 for i in top_5_indices]
    stds = [std_pred[i] * 100 for i in top_5_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    bars = ax.bar(species_names, means, yerr=stds, capsize=5, alpha=0.8, color='skyblue', edgecolor='black')
    ax.set_ylabel('Confidence (%)')
    ax.set_title('Top 5 Predizioni con Intervalli di Incertezza')
    ax.set_ylim(0, 105)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig
    
# --- NUOVO QUIZ DETTAGLIATO ---

# QUIZ ESPANSO
def detailed_questionnaire():
    st.subheader("📋 Valutazione Dettagliata")
    # Dimensioni di Fiducia
    competence_trust = st.slider("Quanto ti sembra **competente** il sistema in questo compito? (1=Per niente, 7=Moltissimo)", 1, 7, 4, key="competence")
    benevolence_trust = st.slider("Pensi che il sistema sia progettato per **aiutare l'utente**? (Benevolenza) (1-7)", 1, 7, 4, key="benevolence")
    integrity_trust = st.slider("Pensi che il sistema sia **onesto** riguardo la sua incertezza? (Integrità) (1-7)", 1, 7, 4, key="integrity")
    # Qualità della Spiegazione e Decisione
    usefulness = st.slider("Quanto ti è stata **utile** la spiegazione per prendere la tua decisione? (1-7)", 1, 7, 4, key="usefulness")
    decision_confidence = st.slider("Quanto sei **sicuro/a** della tua decisione finale dopo aver visto la spiegazione? (1-7)", 1, 7, 4, key="decision_conf")
    
    return {
        'Competenza_Sistema': competence_trust,
        'Benevolenza_Sistema': benevolence_trust,
        'Integrita_Sistema': integrity_trust,
        'Utilità_Spiegazione': usefulness,
        'Sicurezza_Decisione': decision_confidence
    }


# --- FUNZIONI ESISTENTI (invariate ma necessarie) ---

@st.cache_resource
def load_model():
    model_url = 'https://www.dropbox.com/scl/fi/437k0jr5hvzzyfyrp50z2/fungi_classifier_model.h5?rlkey=2tar5m1btexq24y6cf2inosnf&dl=1'
    model_path = 'fungi_classifier_model.h5'
    if not os.path.isfile(model_path):
        with st.spinner(f"Modello non trovato, scaricando..."):
            try:
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except requests.exceptions.RequestException as e:
                st.error(f"Errore durante il download del modello: {e}")
                return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Errore nel caricamento del file del modello: {e}")
        return None

def preprocess_image(image: Image.Image):
    image = image.convert('RGB')
    img_array = np.array(image, dtype=np.uint8)
    img_array = cv2.resize(img_array, (128, 128))
    img_array_scaled = img_array / 255.0
    return np.expand_dims(img_array_scaled, axis=0), img_array

def predict_fungus(model, image_array):
    predictions = model.predict(image_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_species = SPECIES_LIST[predicted_index]
    confidence = predictions[predicted_index] * 100
    info = FUNGI_INFO.get(predicted_species, {"nome_italiano": "N/A", "commestibile": "Sconosciuta"})
    return predicted_species, info, confidence, predictions * 100

def find_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4 and isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def display_superimposed_heatmap(original_image, heatmap, alpha=0.5):
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, original_image, 1 - alpha, 0)
    return superimposed_img

@st.cache_data
def make_gradcam_heatmap(_model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model([_model.inputs], [_model.get_layer(last_conv_layer_name).output, _model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

@st.cache_data
def explain_with_lime(_model, preprocessed_image_array):
    explainer = lime_image.LimeImageExplainer()
    prediction_fn = lambda x: _model.predict(x)
    explanation = explainer.explain_instance(preprocessed_image_array[0], prediction_fn, top_labels=1, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    lime_img = mark_boundaries(temp / 2 + 0.5, mask)
    return lime_img

@st.cache_data
def make_occlusion_sensitivity_map(_model, original_image_resized, patch_size=16):
    original_pred = _model.predict(np.expand_dims(original_image_resized / 255.0, axis=0))[0]
    original_pred_class_prob = np.max(original_pred)
    heatmap = np.zeros((original_image_resized.shape[0], original_image_resized.shape[1]), dtype=np.float32)
    for h in range(0, original_image_resized.shape[0], patch_size):
        for w in range(0, original_image_resized.shape[1], patch_size):
            occluded_image = original_image_resized.copy()
            occluded_image[h:h+patch_size, w:w+patch_size, :] = 0
            occluded_array = np.expand_dims(occluded_image / 255.0, axis=0)
            pred = _model.predict(occluded_array)[0]
            heatmap[h:h+patch_size, w:w+patch_size] = original_pred_class_prob - pred[np.argmax(original_pred)]
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-9)
    return heatmap

@st.cache_data
def fetch_online_images(query: str, num_images: int = 4):
    try:
        api_key = st.secrets["SERPAPI_KEY"]
    except KeyError:
        return "Errore: SERPAPI_KEY non trovata nei secrets di Streamlit."
    params = {"engine": "google_images", "q": query, "api_key": api_key}
    try:
        client = SerpApiClient(params)
        results = client.get_dict()
        return [item['original'] for item in results.get('images_results', [])[:num_images]]
    except Exception as e:
        return f"Errore durante la chiamata API a SerpApi: {e}"

# FUNZIONE DI SALVATAGGIO AGGIORNATA
def save_data_to_google_sheet(data):
    try:
        creds = st.secrets["gcp_service_account"]
        sheet_name = st.secrets["gcp_sheet_name"]
        gc = gspread.service_account_from_dict(creds)
        spreadsheet = gc.open(sheet_name)
        worksheet = spreadsheet.sheet1
        header = [
            "ID_Studente", "Nome_File", "Specie_AI", "Commestibilita_AI",
            "Decisione_Studente", "Fiducia_Generale", "Modalita_Spiegazione",
            "Competenza_Sistema", "Benevolenza_Sistema", "Integrita_Sistema",
            "Utilità_Spiegazione", "Sicurezza_Decisione"
        ]
        ordered_data = [data.get(h) for h in header]
        worksheet.append_row(ordered_data, value_input_option='USER_ENTERED')
        return True, None
    except gspread.exceptions.SpreadsheetNotFound:
        return False, f"Foglio di calcolo '{sheet_name}' non trovato o non condiviso."
    except Exception as e:
        return False, str(e)


# --- INTERFACCIA UTENTE STREAMLIT ---
st.title("🍄 Analisi Funghi con XAI Avanzato")

with st.spinner("Caricamento modello AI..."):
    model = load_model()
if model is None:
    st.error("Impossibile caricare il modello. L'applicazione non può continuare.")
    st.stop()

st.sidebar.header("Impostazioni Esperimento")
uploaded_file = st.sidebar.file_uploader("1. Carica un'immagine di un fungo", type=["jpg", "jpeg", "png"], key="file_uploader")
is_experiment_mode = st.sidebar.checkbox("Attiva Modalità Esperimento", value=True)
student_id = st.sidebar.text_input("ID Studente", "studente_01") if is_experiment_mode else "default"
explanation_mode = st.sidebar.radio("Modalità di Spiegazione", ("Completa (XAI)", "Nessuna (Black Box)")) if is_experiment_mode else "Completa (XAI)"


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    preprocessed_array, original_resized_array = preprocess_image(image)
    predicted_species, info, confidence, all_confidences = predict_fungus(model, preprocessed_array)

    st.header("🎯 Risultati dell'Analisi AI")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption=f"Immagine Caricata", use_container_width=True)
    with col2:
        st.subheader(f"Predizione: **{predicted_species}**")
        st.write(f"*{info.get('nome_italiano', 'N/A')}*")
        
        # NUOVO: Mostra confidenza grezza e calibrata
        factor = CALIBRATION_FACTORS.get(predicted_species, CALIBRATION_FACTORS["default"])
        calibrated_conf = min(confidence * factor, 100)
        st.metric(label="Confidenza Grezza", value=f"{confidence:.2f}%")
        st.metric(label="Confidenza Calibrata", value=f"{calibrated_conf:.2f}%", delta=f"{calibrated_conf - confidence:.2f}%")

        commestibilita = info.get('commestibile', 'Sconosciuta')
        if commestibilita == "Commestibile": st.success(f"**Commestibilità: {commestibilita}** ✅", icon="✅")
        elif commestibilita in ["Velenoso", "Allucinogeno"]: st.warning(f"**Commestibilità: {commestibilita}** ⚠️", icon="⚠️")
        elif commestibilita == "Mortale": st.error(f"**Commestibilità: {commestibilita}** ☠️", icon="☠️")
        else: st.info(f"**Commestibilità: {commestibilita}** ❔", icon="❔")

    with st.expander("Mostra tutte le probabilità di classificazione"):
        st.json({SPECIES_LIST[i]: f"{all_confidences[i]:.2f}%" for i in range(len(SPECIES_LIST))})

    st.divider()

    if explanation_mode == "Completa (XAI)":
        st.header("🤖 Spiegazione della Decisione (XAI)")
        
        tab_list = ["📝 Riepilogo", "🔄 Contrastiva", "📊 Incertezza", "🖼️ Esempi Web", "🔥 Grad-CAM", "🧩 LIME", "⬛ Occlusion"]
        tabs = st.tabs(tab_list)

        with tabs[0]: # Riepilogo
            top_3_indices = np.argsort(all_confidences)[-3:][::-1]
            top_3_preds = [(SPECIES_LIST[i], all_confidences[i]) for i in top_3_indices]
            st.markdown(generate_natural_language_explanation(predicted_species, confidence, top_3_preds))
        with tabs[1]: # Contrastiva
            st.markdown(generate_contrastive_explanation(all_confidences, SPECIES_LIST))
        with tabs[2]: # Incertezza
            with st.spinner("Calcolo incertezza..."):
                mean_pred, std_pred = monte_carlo_uncertainty(model, preprocessed_array)
                st.pyplot(create_uncertainty_visualization(mean_pred, std_pred, SPECIES_LIST))
        with tabs[3]: # Esempi Web
            with st.spinner("Ricerca immagini..."):
                st.image(fetch_online_images(f"{predicted_species} mushroom"), width=150)
        with tabs[4]: # Grad-CAM
            with st.spinner("Generazione Grad-CAM..."):
                last_conv_layer = find_last_conv_layer_name(model)
                st.image(display_superimposed_heatmap(original_resized_array, make_gradcam_heatmap(model, preprocessed_array, last_conv_layer)), use_container_width=True)
        with tabs[5]: # LIME
            with st.spinner("Generazione LIME..."):
                st.image(explain_with_lime(model, preprocessed_array), use_container_width=True)
        with tabs[6]: # Occlusion
            with st.spinner("Generazione Occlusion Map..."):
                st.image(display_superimposed_heatmap(original_resized_array, make_occlusion_sensitivity_map(model, original_resized_array)), use_container_width=True)

    if is_experiment_mode:
        st.divider()
        st.header("🔬 La Tua Valutazione")
        
        final_decision = st.radio(
            "Qual è la tua decisione finale sulla commestibilità?", 
            ("Commestibile", "Non Commestibile / Velenoso", "Non so decidere"), 
            index=None, horizontal=True, key=f"decision_{uploaded_file.name}"
        )
        
        detailed_responses = detailed_questionnaire()
        
        trust_rating = st.slider(
            "Complessivamente, quanta fiducia riponi nel sistema AI? (1=Nessuna, 7=Massima)", 
            1, 7, 4, key=f"trust_{uploaded_file.name}"
        )

        if st.button("Salva e Invia la mia Decisione"):
            if final_decision and student_id:
                experiment_data = {
                    "ID_Studente": student_id, "Nome_File": uploaded_file.name,
                    "Specie_AI": predicted_species, "Commestibilita_AI": commestibilita,
                    "Decisione_Studente": final_decision, "Fiducia_Generale": trust_rating,
                    "Modalita_Spiegazione": explanation_mode
                }
                experiment_data.update(detailed_responses)
                success, error_message = save_data_to_google_sheet(experiment_data)
                
                if success:
                    st.success("Decisione registrata con successo su Google Sheet! Grazie.")
                else:
                    st.error(f"Errore durante il salvataggio: {error_message}")
            else:
                st.error("Per favore, compila l'ID studente e fai una scelta prima di salvare.")
