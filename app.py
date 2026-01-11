import streamlit as st
import pandas as pd
import os
import json
import sys
import time
import glob
import signal
import threading
import io
import zipfile
from datetime import datetime

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.graph.graph import app_graph, request_abort, clear_abort

_SIGNAL_HANDLER_INSTALLED = False

def _load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _handle_shutdown(signum, frame):
    request_abort(f"signal={signum}")
    raise KeyboardInterrupt

def _install_signal_handlers():
    global _SIGNAL_HANDLER_INSTALLED
    if _SIGNAL_HANDLER_INSTALLED:
        return
    if threading.current_thread() is not threading.main_thread():
        return
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)
    _SIGNAL_HANDLER_INSTALLED = True

_install_signal_handlers()

# 1. Configuración Visual
st.set_page_config(
    page_title="The Insight Foundry",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏭 The Insight Foundry")
st.markdown("### 🤖 Sistema Multi-Agente de Inteligencia de Negocio")
st.markdown("---")

# 2. Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    st.info("Sube tus datos y define tu objetivo.")
    
    uploaded_file = st.file_uploader("📂 Cargar CSV", type=["csv"])
    
    business_objective = st.text_area(
        "🎯 Objetivo de Negocio",
        placeholder="Ej: Reducir el churn de clientes en un 10%...",
        height=150
    )
    
    start_btn = st.button("🚀 Iniciar Análisis")

# Main Logic
if "analysis_complete" not in st.session_state:
    st.session_state["analysis_complete"] = False
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None

# Main Logic
if uploaded_file is not None:
    # Save file
    os.makedirs("data", exist_ok=True)
    data_path = os.path.join("data", uploaded_file.name)
    with open(data_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Preview logic (only if not analyzing)
    if not st.session_state["analysis_complete"] and not start_btn:
        st.subheader("📊 Vista Previa de Datos")
        
        def load_preview(file):
            # Intento 1: Estándar
            file.seek(0)
            try:
                df = pd.read_csv(file)
                if len(df.columns) > 1:
                    return df
            except Exception:
                pass
            
            # Intento 2: Punto y coma
            file.seek(0)
            try:
                df = pd.read_csv(file, sep=';')
                if len(df.columns) > 1:
                    return df
            except Exception:
                pass
                
            # Intento 3: Latin-1 y Punto y coma
            file.seek(0)
            try:
                df = pd.read_csv(file, sep=';', encoding='latin-1')
                return df
            except Exception:
                return None

        df_preview = load_preview(uploaded_file)
        
        if df_preview is not None:
            st.dataframe(df_preview.head(), width="stretch")
        else:
            st.error("Error al leer CSV: No se pudo detectar el formato automáticamente.")

# START BUTTON LOGIC
if start_btn:
    if uploaded_file is None:
        st.sidebar.error("⚠️ Por favor sube un archivo CSV.")
    elif not business_objective:
        st.sidebar.error("⚠️ Por favor define un objetivo.")
    else:
        # Reset state for new run
        st.session_state["analysis_complete"] = False
        st.session_state["analysis_result"] = None
        clear_abort()
        
        # Clean previous plots
        if os.path.exists("static/plots"):
            files = glob.glob("static/plots/*")
            for f in files:
                os.remove(f)

        # 3. Ejecución con Feedback
        try:
            with st.status("🚀 Agentes IA trabajando en tu caso...", expanded=True) as status:
                st.write("🕵️ **Data Steward:** Analizando calidad e integridad de datos...")
                
                initial_state = {
                    "csv_path": data_path,
                    "business_objective": business_objective
                }
                
                # Streaming Execution
                final_state = initial_state.copy()
                
                for event in app_graph.stream(initial_state, config={"recursion_limit": 100}):
                    if event is None:
                        continue
                        
                    # Update final_state with new data from event
                    for key, value in event.items():
                        if value is not None:
                            final_state.update(value)
                    
                    # Status Updates based on active node
                    if 'steward' in event:
                        st.write("✅ Datos auditados.")
                        st.write("🧠 **Strategist:** Diseñando estrategias de alto impacto...")
                    
                    elif 'strategist' in event:
                        strategies_payload = final_state.get('strategies', {})
                        st.write(f"🧠 **Strategist:** 3 Estrategias generadas. Iniciando deliberación...")
                        
                    elif 'domain_expert' in event:
                        selected = final_state.get('selected_strategy', {})
                        reason = final_state.get('selection_reason', '')
                        st.success(f"🏆 **Domain Expert:** Estrategia ganadora seleccionada: {selected.get('title')}")
                        st.info(f"Razón: {reason}")
                        st.write("🧹 **Data Engineer:** Limpiando y estandarizando dataset...")

                    # Selector removed
                    # elif 'selector' in event: ...

                    elif 'data_engineer' in event:
                        st.write("✅ Datos limpiados y estandarizados.")
                        st.write("⚙️ **ML Engineer:** Optimizando modelo (Iteración en curso)...")
                    
                    elif 'engineer' in event:
                        pass # Keep "Optimizando modelo..." message
                        
                    elif 'execute_code' in event:
                        st.write("✅ Ejecución de código finalizada.")
                        st.write("🧐 **Reviewer:** Evaluando resultados de negocio...")

                    elif 'evaluate_results' in event:
                         verdict = final_state.get('review_verdict', 'APPROVED')
                         if verdict == "NEEDS_IMPROVEMENT":
                             feedback = final_state.get('execution_feedback', '')
                             st.warning(f"⚠️ Resultados insuficientes. Reviewer solicita mejoras: {feedback}")
                             st.write("⚙️ **ML Engineer:** Refinando estrategia (Retry)...")
                         else:
                             st.success("✅ Resultados Aprobados por Negocio.")
                    
                    elif 'retry_handler' in event:
                        pass # Feedback handled above
                        
                    elif 'translator' in event:
                        st.write("📊 **Traductor:** Generando reporte ejecutivo...")

                    elif 'generate_pdf' in event:
                        st.write("📄 **Sistema:** Generando PDF final...")
                
                # STORE RESULT IN SESSION STATE
                st.session_state["analysis_result"] = final_state
                st.session_state["analysis_complete"] = True
                
                status.update(label="✅ ¡Análisis Completado!", state="complete", expanded=False)
                
                # Force Rerun to render results in the main flow
                st.rerun()
            
        except Exception as e:
            st.error(f"Ocurrió un error crítico: {e}")
            st.exception(e)

# RENDER RESULTS FROM STATE (PERSISTENT AFTER RELOAD)
if st.session_state.get("analysis_complete") and st.session_state.get("analysis_result"):
    result = st.session_state["analysis_result"]
    
    # Debugging Visual
    with st.sidebar:
         st.write("Debug State Keys:", list(result.keys()))
    
    # 4. Visualización con Tabs
    st.balloons()
    
    tab1, tab2, tab_de, tab3, tab4 = st.tabs([
        "🧐 Data Audit", 
        "🧠 Estrategia", 
        "🧹 Ingeniería de Datos",
        "🤖 ML Engineer", 
        "💼 Reporte Ejecutivo"
    ])
    
    with tab1:
        st.subheader("Auditoría de Datos (Steward)")
        st.info(result.get('data_summary', 'No disponible'))
    
    with tab2:
        st.subheader("Plan Estratégico (Strategist)")
        strategies = result.get('strategies', {})
        
        if isinstance(strategies, dict) and 'strategies' in strategies:
            for i, strat in enumerate(strategies['strategies'], 1):
                with st.expander(f"Estrategia {i}: {strat.get('title')}", expanded=True):
                    st.write(f"**Hipótesis:** {strat.get('hypothesis')}")
                    st.write(f"**Dificultad:** {strat.get('estimated_difficulty')}")
                    st.write(f"**Razonamiento:** {strat.get('reasoning')}")
        else:
            st.json(strategies)
            
        selected = result.get('selected_strategy', {})
        reviews = result.get('domain_expert_reviews', [])

        if selected:
            st.divider()
            st.success(f"🏆 **Estrategia Ganadora:** {selected.get('title')}")
            st.write(f"**Motivo de Selección:** {result.get('selection_reason', 'N/A')}")
        
        if reviews:
            st.subheader("🧐 Deliberación del Experto (Scores)")
            for rev in reviews:
                with st.expander(f"Evaluación: {rev.get('title')} (Score: {rev.get('score')}/10)"):
                    st.write(f"**Reasoning:** {rev.get('reasoning')}")
                    st.write(f"**Risks:** {rev.get('risks')}")
                    st.write(f"**Recommendation:** {rev.get('recommendation')}")

    with tab_de:
        st.subheader("Ingeniería de Datos (Data Engineer)")
        
        code = result.get('cleaning_code', '# No code available')
        preview = result.get('cleaned_data_preview', 'No preview available')
        
        col_de_code, col_de_preview = st.columns(2)
        
        with col_de_code:
            st.markdown("**Script de Limpieza Generado:**")
            st.code(code, language='python')
            
        with col_de_preview:
            st.markdown("**Vista Previa (Cleaned Data):**")
            if isinstance(preview, str) and preview.strip().startswith('{'):
                 try:
                     from io import StringIO
                     st.dataframe(pd.read_json(StringIO(preview), orient='split'), width="stretch")
                 except Exception as e:
                     st.write(f"Cannot render dataframe: {e}")
                     st.write(preview)
            else:
                 st.write(preview)

    with tab3:
        st.subheader("Ejecución Técnica (ML Engineer)")
        
        col_code, col_out = st.columns(2)
        
        with col_code:
            st.markdown("**Código Generado:**")
            ml_code = result.get('generated_code', '# No code')
            if ml_code.strip() == "# Generation Failed":
                ml_code = result.get('last_generated_code', ml_code)
            st.code(ml_code, language='python')
        
        with col_out:
            st.markdown("**Salida de Consola:**")
            ml_output = result.get('execution_output', '')
            last_ok = result.get('last_successful_execution_output')
            if "BUDGET_EXCEEDED" in str(ml_output) and last_ok:
                ml_output = f"{ml_output}\n\n--- Last successful execution output ---\n{last_ok}"
            st.text_area("Output", ml_output, height=400)

    with tab4:
        st.subheader("Informe para Directivos (Translator)")
        st.markdown(result.get('final_report', 'No disponible'))
        
        # 📊 Visualización de Datos
        plots = glob.glob("static/plots/*.png")
        if plots:
            st.markdown("### 📊 Visualización de Datos")
            cols = st.columns(2)
            for i, plot_path in enumerate(plots):
                with cols[i % 2]:
                    st.image(plot_path, caption=os.path.basename(plot_path), use_column_width=True)
        
        # PDF Download Button - Robust Persistence
        # 1. Check if binary is already in memory
        if 'pdf_binary' not in st.session_state:
             # 2. If not, try to recover it from the result path (e.g., after refresh)
             pdf_path = result.get('pdf_path')
             if pdf_path and os.path.exists(pdf_path):
                 try:
                     with open(pdf_path, "rb") as pdf_file:
                         st.session_state['pdf_binary'] = pdf_file.read()
                 except Exception as e:
                     st.warning(f"Could not reload PDF: {e}")

        # 3. Render button if binary is available
        if 'pdf_binary' in st.session_state:
            st.markdown("---")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            st.download_button(
                label="📄 Descargar Reporte PDF",
                data=st.session_state['pdf_binary'],
                file_name=f"Reporte_Ejecutivo_{timestamp}.pdf",
                mime="application/pdf"
            )

        # Download ML artifacts based on outputs generated in this run
        output_report = result.get("output_contract_report")
        if not isinstance(output_report, dict):
            output_report = _load_json("data/output_contract_report.json") or {}
        present_outputs = output_report.get("present", []) if isinstance(output_report, dict) else []
        present_files = [p for p in present_outputs if isinstance(p, str) and os.path.exists(p)]

        if present_files:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in present_files:
                    arcname = os.path.relpath(file_path, start=os.getcwd())
                    zf.write(file_path, arcname=arcname)
            zip_buffer.seek(0)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            st.download_button(
                label="Descargar entregables ML (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"Entregables_ML_{timestamp}.zip",
                mime="application/zip"
            )
        else:
            st.info("No se encontraron entregables ML para descargar en esta ejecución.")
