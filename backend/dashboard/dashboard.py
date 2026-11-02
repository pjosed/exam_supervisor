"""
Dashboard en tiempo real para el profesor.
Lee el estado del sistema desde un archivo JSON compartido.
"""

import json
import time
from pathlib import Path

import streamlit as st

STATE_FILE = Path("runtime_state.json")
REFRESH_INTERVAL = 1  # segundos

st.set_page_config(
    page_title="Exam Supervisor",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ Exam Supervisor — Dashboard del Profesor")
st.caption("Actualización automática cada segundo")

placeholder = st.empty()

while True:
    with placeholder.container():
        if not STATE_FILE.exists():
            st.warning("⏳ Esperando que el sistema de cámara inicie...")
        else:
            try:
                data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                persons = data.get("persons", [])
                timestamp = data.get("timestamp", "")

                st.caption(f"Última actualización: {timestamp}")

                if not persons:
                    st.info("No se detectan personas en cámara.")
                else:
                    cols = st.columns(min(len(persons), 4))
                    for i, person in enumerate(persons):
                        col = cols[i % 4]
                        level = person.get("level", "GREEN")
                        score = person.get("score", 0)
                        tid   = person.get("track_id", "?")

                        color_map = {
                            "GREEN":  "🟢",
                            "YELLOW": "🟡",
                            "RED":    "🔴"
                        }
                        bg_map = {
                            "GREEN":  "#1a3a1a",
                            "YELLOW": "#3a3a1a",
                            "RED":    "#3a1a1a"
                        }

                        icon = color_map.get(level, "⚪")
                        bg   = bg_map.get(level, "#1a1a1a")

                        with col:
                            st.markdown(f"""
                            <div style="background:{bg};padding:16px;border-radius:10px;margin:4px">
                                <h3 style="color:white">{icon} Estudiante {tid}</h3>
                                <h2 style="color:white">{score:.0f} pts</h2>
                                <p style="color:#aaa">Nivel: {level}</p>
                            </div>
                            """, unsafe_allow_html=True)

                            alerts = person.get("alerts", [])
                            if alerts:
                                for alert in alerts:
                                    st.warning(alert)

                # Alertas rojas al tope
                red_persons = [p for p in persons if p.get("level") == "RED"]
                if red_persons:
                    st.error(f"⚠️ {len(red_persons)} estudiante(s) requieren revisión humana")

            except Exception as e:
                st.error(f"Error leyendo estado: {e}")

    time.sleep(REFRESH_INTERVAL)
    st.rerun()