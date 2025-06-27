import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
from fpdf import FPDF
import re
import fitz  # PyMuPDF
from typing import Dict, List, Optional, Tuple
import logging
import tempfile
import shutil
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Configuración inicial
st.set_page_config(
    page_title="GEOTEC - Sistema Experto de Estabilización de Suelos",
    layout="wide",
    page_icon="🌍",
    initial_sidebar_state="expanded"
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Constantes
MAX_TOKENS = 4000
TEMP_DIR = "temp_reports"
os.makedirs(TEMP_DIR, exist_ok=True)
NORMATIVES_DIR = "normas"
ARTICLES_DIR = "articulos"

# Tipos de suelos completos según USCS
SOIL_TYPES = [
    "Arcilla", "Arcilla limosa", "Arcilla orgánica",
    "Limo", "Limo orgánica", "Arena", "Arena limosa",
    "Arena arcillosa", "Grava", "Grava limosa", 
    "Grava arcillosa", "Suelo orgánico", "Turba",
    "Loess", "Laterita", "Bentonita", "Margas", "Arcillas expansivas",
    "Suelos colapsables", "Suelos residuales", "Suelos aluviales"
]

# Configuración de búsqueda académica
SEARCH_ENGINES = {
    "Google Scholar": "https://scholar.google.com/scholar?q=",
    "ScienceDirect": "https://www.sciencedirect.com/search?qs=",
    "ResearchGate": "https://www.researchgate.net/search?q=",
    "SpringerLink": "https://link.springer.com/search?query=",
    "ASCE Library": "https://ascelibrary.org/action/doSearch?AllField="
}

class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)
        
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Informe Técnico de Estabilización de Suelos', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')
    
    def add_section(self, title, content):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, title, 0, 1)
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, content)
        self.ln(5)

def load_documents(folder: str) -> Dict[str, str]:
    """Carga documentos PDF desde una carpeta"""
    documents = {}
    if not os.path.exists(folder):
        logger.warning(f"Carpeta {folder} no encontrada")
        return documents
    
    for file in os.listdir(folder):
        if file.lower().endswith(".pdf"):
            try:
                with fitz.open(os.path.join(folder, file)) as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    documents[file] = text[:5000]  # Limitar a 5000 caracteres
            except Exception as e:
                logger.error(f"Error leyendo {file}: {str(e)}")
    return documents

def search_academic_references(query: str) -> List[Dict]:
    """Busca referencias académicas relevantes"""
    results = []
    try:
        url = f"{SEARCH_ENGINES['Google Scholar']}{query.replace(' ', '+')}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for item in soup.select(".gs_ri")[:5]:  # Aumentamos a 5 resultados
            title = item.select_one(".gs_rt").get_text()
            authors_source = item.select_one(".gs_a").get_text()
            snippet = item.select_one(".gs_rs").get_text()
            link = item.select_one(".gs_rt a")["href"] if item.select_one(".gs_rt a") else None
            
            parts = authors_source.split(" - ")
            authors = parts[0] if len(parts) > 0 else "Desconocido"
            source = parts[1] if len(parts) > 1 else "Desconocido"
            year = re.search(r"\b(19|20)\d{2}\b", authors_source)
            year = year.group() if year else "Desconocido"
            
            results.append({
                "title": title,
                "authors": authors,
                "source": source,
                "year": year,
                "snippet": snippet,
                "url": link,
                "engine": "Google Scholar"
            })
    
    except Exception as e:
        logger.error(f"Error en búsqueda académica: {str(e)}")
    
    return results

def validate_soil_parameters(data: pd.Series) -> Optional[str]:
    """Valida la coherencia de los parámetros del suelo"""
    errors = []
    
    # Validación de niveles freáticos
    if data['Nivel freático (m)'] < 0:
        errors.append("El nivel freático no puede ser negativo")
    
    # Validación de presión de carga
    if data['Presión de carga (kPa)'] < 0:
        errors.append("La presión de carga no puede ser negativa")
    
    # Validación de límites de Atterberg solo si se proporcionaron ambos
    if 'Límite líquido (LL)' in data and 'Límite plástico (LP)' in data:
        if data['Límite líquido (LL)'] < data['Límite plástico (LP)']:
            errors.append("El límite líquido (LL) no puede ser menor que el límite plástico (LP)")
    
    # Validación de granulometría solo si se proporcionó
    if all(key in data for key in ['Grava (%)', 'Arena (%)', 'Limo (%)', 'Arcilla (%)']):
        if data['Grava (%)'] + data['Arena (%)'] + data['Limo (%)'] + data['Arcilla (%)'] != 100:
            errors.append("La suma de los porcentajes de granulometría debe ser 100%")
    
    if errors:
        return "\n".join(f"- {error}" for error in errors)
    return None

def data_input_interface() -> Tuple[Optional[pd.DataFrame], str]:
    """Interfaz de entrada de datos con parámetros opcionales"""
    with st.form("soil_data_form"):
        st.header("Datos del Suelo y Proyecto")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Parámetros obligatorios
            soil_type = st.selectbox(
                "Tipo de suelo*",
                SOIL_TYPES,
                help="Seleccione según clasificación USCS"
            )
            water_level = st.number_input(
                "Nivel freático (metros)*", 
                min_value=0.0,
                value=1.5,
                step=0.1,
                format="%.1f"
            )
            
        with col2:
            # Parámetros obligatorios
            load_pressure = st.number_input(
                "Presión de carga (kPa)*", 
                min_value=0,
                value=150,
                step=10
            )
            desired_strength = st.number_input(
                "Resistencia deseada (kPa)", 
                min_value=0,
                value=200
            )
        
        # Sección expandible para parámetros opcionales
        with st.expander("Parámetros adicionales (opcionales)", expanded=False):
            st.subheader("Propiedades del Suelo")
            
            # Granulometría
            st.markdown("**Granulometría (%):**")
            gravel_col, sand_col, silt_col, clay_col = st.columns(4)
            with gravel_col:
                gravel = st.number_input("Grava (%)", min_value=0, max_value=100, value=0)
            with sand_col:
                sand = st.number_input("Arena (%)", min_value=0, max_value=100, value=0)
            with silt_col:
                silt = st.number_input("Limo (%)", min_value=0, max_value=100, value=0)
            with clay_col:
                clay = st.number_input("Arcilla (%)", min_value=0, max_value=100, value=0)
            
            # Límites de Atterberg
            st.markdown("**Límites de Atterberg:**")
            ll_col, lp_col = st.columns(2)
            with ll_col:
                ll = st.number_input("Límite líquido (LL)", min_value=0, value=0)
            with lp_col:
                lp = st.number_input("Límite plástico (LP)", min_value=0, value=0)
            
            # IP se calcula automáticamente si se proporcionan LL y LP
            if ll > 0 and lp > 0:
                ip = ll - lp
                st.markdown(f"**Índice de plasticidad (IP):** {ip} (calculado automáticamente como LL - LP)")
            else:
                ip = None
            
            # Parámetros adicionales
            st.subheader("Otras Propiedades")
            moisture_col, ph_col = st.columns(2)
            with moisture_col:
                moisture_content = st.number_input(
                    "Contenido de humedad natural (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.5,
                    format="%.1f"
                )
            with ph_col:
                ph_value = st.number_input(
                    "pH del suelo",
                    min_value=0.0,
                    max_value=14.0,
                    value=0.0,
                    step=0.1,
                    format="%.1f"
                )
            
            cbr_col, swelling_col = st.columns(2)
            with cbr_col:
                cbr = st.number_input(
                    "CBR (%)",
                    min_value=0,
                    max_value=100,
                    value=0,
                    help="Valor de California Bearing Ratio"
                )
            with swelling_col:
                swelling = st.number_input(
                    "Potencial de hinchamiento (%)",
                    min_value=0.0,
                    value=0.0,
                    step=0.5,
                    format="%.1f"
                )
        
        additional_info = st.text_area(
            "Información adicional del proyecto:",
            height=100,
            placeholder="Descripción del proyecto, condiciones especiales, restricciones..."
        )
        
        if st.form_submit_button("Analizar y Recomendar"):
            # Construir diccionario de datos solo con valores proporcionados
            data = {
                "Tipo de suelo": soil_type,
                "Nivel freático (m)": water_level,
                "Presión de carga (kPa)": load_pressure,
                "Resistencia deseada (kPa)": desired_strength if desired_strength > 0 else None
            }
            
            # Agregar parámetros opcionales solo si tienen valores
            if gravel + sand + silt + clay > 0:
                data.update({
                    "Grava (%)": gravel,
                    "Arena (%)": sand,
                    "Limo (%)": silt,
                    "Arcilla (%)": clay
                })
            
            if ll > 0:
                data["Límite líquido (LL)"] = ll
            if lp > 0:
                data["Límite plástico (LP)"] = lp
            if ip is not None:
                data["Índice de plasticidad (IP)"] = ip
            
            if moisture_content > 0:
                data["Contenido de humedad (%)"] = moisture_content
            if ph_value > 0:
                data["pH del suelo"] = ph_value
            if cbr > 0:
                data["CBR (%)"] = cbr
            if swelling > 0:
                data["Potencial de hinchamiento (%)"] = swelling
            
            # Validar parámetros
            validation_error = validate_soil_parameters(pd.Series(data))
            if validation_error:
                st.error(f"Error en los parámetros ingresados:\n{validation_error}")
                return None, ""
            
            return pd.DataFrame([data]), additional_info
    
    return None, ""

def generate_technical_prompt(data: pd.Series, additional_info: str, normatives: Dict[str, str], articles: Dict[str, str]) -> str:
    """Genera un prompt técnico detallado con contexto de normativas y artículos"""
    # Búsqueda de referencias académicas específicas
    search_queries = [
        f"estabilización de {data['Tipo de suelo']} nivel freático {data['Nivel freático (m)']}m",
        f"{data['Tipo de suelo']} presión de carga {data['Presión de carga (kPa)']}kPa"
    ]
    
    # Agregar búsquedas específicas según parámetros disponibles
    if 'Índice de plasticidad (IP)' in data:
        search_queries.append(f"métodos de estabilización para {data['Tipo de suelo']} IP{data['Índice de plasticidad (IP)']}")
    elif 'Límite líquido (LL)' in data:
        search_queries.append(f"métodos de estabilización para {data['Tipo de suelo']} LL{data['Límite líquido (LL)']}")
    
    academic_refs = []
    for query in search_queries:
        academic_refs.extend(search_academic_references(query))
    
    # Eliminar duplicados
    unique_refs = []
    seen_titles = set()
    for ref in academic_refs:
        if ref['title'] not in seen_titles:
            seen_titles.add(ref['title'])
            unique_refs.append(ref)
    
    # Formatear referencias académicas
    refs_text = "\n".join(
        f"- {ref['authors']} ({ref['year']}). {ref['title']}. {ref['source']} [URL: {ref['url']}]"
        for ref in unique_refs[:10]  # Limitar a 10 referencias
    ) if unique_refs else "No se encontraron referencias adicionales"
    
    # Contexto de normativas
    normative_context = "\nNormativas relevantes:\n" + "\n".join(
        f"- {name}: {content[:500]}..." 
        for name, content in list(normatives.items())[:5]
    ) if normatives else "\nNo hay normativas cargadas"
    
    # Contexto de artículos
    article_context = "\nArtículos técnicos:\n" + "\n".join(
        f"- {name}: {content[:500]}..." 
        for name, content in list(articles.items())[:5]
    ) if articles else "\nNo hay artículos cargados"
    
    # Construir descripción de datos técnicos
    soil_data_text = f"""
    ### Datos Técnicos del Suelo:
    1. Tipo de suelo: {data['Tipo de suelo']}
    2. Nivel freático: {data['Nivel freático (m)']} m
    3. Presión de carga: {data['Presión de carga (kPa)']} kPa"""
    
    if 'Resistencia deseada (kPa)' in data and data['Resistencia deseada (kPa)'] is not None:
        soil_data_text += f"\n4. Resistencia deseada: {data['Resistencia deseada (kPa)']} kPa"
    
    if all(key in data for key in ['Grava (%)', 'Arena (%)', 'Limo (%)', 'Arcilla (%)']):
        soil_data_text += f"""
    5. Granulometría:
       - Grava: {data['Grava (%)']}%
       - Arena: {data['Arena (%)']}%
       - Limo: {data['Limo (%)']}%
       - Arcilla: {data['Arcilla (%)']}%"""
    
    if 'Límite líquido (LL)' in data and 'Límite plástico (LP)' in data:
        soil_data_text += f"""
    6. Límites de Atterberg:
       - Límite líquido (LL): {data['Límite líquido (LL)']}
       - Límite plástico (LP): {data['Límite plástico (LP)']}"""
        if 'Índice de plasticidad (IP)' in data:
            soil_data_text += f"\n       - Índice de plasticidad (IP): {data['Índice de plasticidad (IP)']}"
    
    if 'Contenido de humedad (%)' in data:
        soil_data_text += f"\n7. Contenido de humedad natural: {data['Contenido de humedad (%)']}%"
    
    if 'pH del suelo' in data:
        soil_data_text += f"\n8. pH del suelo: {data['pH del suelo']}"
    
    if 'CBR (%)' in data:
        soil_data_text += f"\n9. CBR: {data['CBR (%)']}%"
    
    if 'Potencial de hinchamiento (%)' in data:
        soil_data_text += f"\n10. Potencial de hinchamiento: {data['Potencial de hinchamiento (%)']}%"
    
    return f"""
    Eres un ingeniero geotécnico senior con 30 años de experiencia en estabilización de suelos. 
    Realiza un análisis exhaustivo para este caso específico:

    {soil_data_text}

    ### Contexto Técnico:
    {normative_context}
    {article_context}

    ### Referencias Académicas Relevantes:
    {refs_text}

    ### Información Adicional del Proyecto:
    {additional_info if additional_info else 'Ninguna'}

    ### Requerimientos del Análisis:
    1. **Evaluación de Parámetros:**
       - Verifica la coherencia de los parámetros ingresados
       - Si hay inconsistencias, explica por qué no se puede realizar el análisis
       - Considera que algunos parámetros pueden no estar disponibles

    2. **Clasificación Detallada:**
       - Clasifica el suelo según USCS y AASHTO con los datos disponibles
       - Explica cada parámetro disponible y su implicación
       - Indica las limitaciones por falta de datos si es necesario

    3. **Problemas Identificados:**
       - Lista los problemas específicos para este suelo
       - Relaciona cada problema con los parámetros ingresados
       - Señala posibles problemas no identificables por falta de datos

    4. **Recomendación de Estabilización:**
       - Realiza un análisis exhaustivo considerando:
         * Métodos físicos (compactación, inclusión de geosintéticos)
         * Métodos químicos (cal, cemento, polímeros)
         * Métodos innovadores (biocementación, nanotecnología)
       - Selecciona UN único método óptimo basado en los datos disponibles
       - El método recomendado debe ser SUPER ESPECÍFICO (no genérico)
       - Considera que algunos métodos pueden no ser evaluables por falta de datos

    5. **Justificación Técnica Rigurosa:**
       - Explica el mecanismo de acción del método recomendado
       - Detalla materiales requeridos con especificaciones técnicas
       - Describe el proceso constructivo paso a paso
       - Presenta resultados esperados cuantificables
       - Cita normativas aplicables (ASTM, AASHTO, ISO, etc) con números exactos
       - Referencia artículos técnicos que respalden la recomendación
       - Compara con otros métodos descartados y explica por qué no son óptimos
       - Indica cualquier limitación en el análisis debido a falta de datos

    6. **Aplicaciones Recomendadas:**
       - Proporciona aplicaciones específicas basadas en los datos disponibles con:
         * Tipo de proyecto exacto (ej: "Cimentación para edificio de 5 pisos")
         * Configuración recomendada
         * Ejemplos reales documentados (si existen)
         * Justificación técnica para cada aplicación
       - Indica si las recomendaciones podrían refinarse con datos adicionales

    ### Formato de Respuesta Estricto:
    **Evaluación de Parámetros:**
    [Análisis de coherencia de los datos ingresados y limitaciones por datos faltantes]

    **Clasificación del Suelo:**
    [Clasificación detallada según sistemas estándar con los datos disponibles]

    **Problemas Identificados:**
    [Lista de problemas específicos para este suelo y posibles riesgos no evaluables]

    **Recomendación Óptima:**
    [Método específico recomendado con consideración de datos faltantes]

    **Justificación Técnica:**
    [Explicación detallada con fundamentos técnicos, normativas y referencias]

    **Aplicaciones Recomendadas:**
    1. [Proyecto específico 1 con justificación]
    2. [Proyecto específico 2 con justificación]
    3. [Proyecto específico 3 con justificación]
    4. [Proyecto específico 4 con justificación]
    5. [Proyecto específico 5 con justificación]
    """

# [Las funciones restantes (query_ai, parse_response, display_results, generate_pdf_report, main) permanecen exactamente iguales que en el código anterior]

def query_ai(prompt: str) -> str:
    """Consulta a la API de OpenAI con enfoque técnico riguroso"""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un ingeniero geotécnico senior con 30 años de experiencia en estabilización de suelos. "
                        "Realiza análisis técnicos exhaustivos basados en evidencia científica y normativa. "
                        "Sigue estrictamente estos requisitos:\n"
                        "1. Evalúa primero la coherencia de los parámetros ingresados\n"
                        "2. Clasifica el suelo con precisión según los estándares con los datos disponibles\n"
                        "3. Identifica problemas específicos basados en los datos proporcionados\n"
                        "4. Recomienda UN único método ESPECÍFICO después de analizar todas las opciones\n"
                        "5. Justifica con normativas exactas (ASTM, AASHTO, ISO) y artículos científicos indexados\n"
                        "6. Propone aplicaciones específicas con ejemplos reales cuando sea posible\n"
                        "7. Indica claramente cualquier limitación debido a datos faltantes\n"
                        "Sé extremadamente preciso y técnico en todas las explicaciones."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Baja temperatura para mayor precisión
            max_tokens=MAX_TOKENS,
            top_p=0.9  # Para mayor diversidad en las recomendaciones
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Error en consulta al modelo: {str(e)}")
        return f"Error: {str(e)}"

def parse_response(content: str) -> Dict[str, str]:
    """Parsea la respuesta en secciones estructuradas"""
    sections = {
        "parameter_evaluation": "",
        "classification": "",
        "problems": "",
        "recommendation": "",
        "justification": "",
        "applications": "",
        "norms": [],
        "references": []
    }
    
    # Extraer normativas citadas
    norms = re.findall(
        r"(ASTM [A-Z]+\s?\d+|AASHTO [A-Z]+\s?\d+|ISO \d+-\d+|EN \d+|NTC \d+)", 
        content, re.IGNORECASE
    )
    sections["norms"] = list(set(norms))
    
    # Extraer referencias bibliográficas
    ref_pattern = re.compile(
        r"(?P<authors>[A-Za-zÁ-ÿ\s\.,]+(?:et al\.)?)\s*\((?P<year>\d{4})\)[^.]*\.\s*(?P<title>[^.]*?)\s*\.\s*(?P<source>[^.]*?)(?:\.|$)"
    )
    sections["references"] = [
        {
            "authors": match.group("authors").strip(),
            "year": match.group("year"),
            "title": match.group("title").strip(),
            "source": match.group("source").strip()
        }
        for match in ref_pattern.finditer(content)
    ]
    
    # Extraer secciones principales
    section_markers = {
        "parameter_evaluation": "Evaluación de Parámetros:",
        "classification": "Clasificación del Suelo:",
        "problems": "Problemas Identificados:",
        "recommendation": "Recomendación Óptima:",
        "justification": "Justificación Técnica:",
        "applications": "Aplicaciones Recomendadas:"
    }
    
    for section, marker in section_markers.items():
        if marker in content:
            end_marker = next((m for m in section_markers.values() 
                             if m != marker and content.find(m) > content.find(marker)), None)
            
            section_content = content.split(marker)[1]
            if end_marker:
                section_content = section_content.split(end_marker)[0]
            
            sections[section] = section_content.strip()
    
    return sections

def display_results(data: pd.Series, analysis: str):
    """Muestra los resultados de forma estructurada y profesional"""
    # Parsear la respuesta
    sections = parse_response(analysis)
    
    # Mostrar evaluación de parámetros primero
    if sections["parameter_evaluation"]:
        if "inconsistencias" in sections["parameter_evaluation"].lower() or "error" in sections["parameter_evaluation"].lower():
            st.error("Problemas con los parámetros ingresados:")
            st.markdown(sections["parameter_evaluation"])
            return
    
    st.success("Análisis completado exitosamente")
    
    # Mostrar en pestañas
    tab1, tab2, tab3 = st.tabs(["Resumen Técnico", "Recomendación Detallada", "Aplicaciones Específicas"])
    
    with tab1:
        st.subheader("Clasificación del Suelo")
        if sections["classification"]:
            st.markdown(sections["classification"])
        else:
            st.warning("No se pudo determinar la clasificación del suelo")
        
        st.subheader("Problemas Identificados")
        if sections["problems"]:
            st.markdown(sections["problems"])
        else:
            st.warning("No se identificaron problemas específicos")
    
    with tab2:
        st.subheader("Método de Estabilización Recomendado")
        if sections["recommendation"]:
            st.markdown(sections["recommendation"], unsafe_allow_html=True)
            
            st.subheader("Justificación Técnica")
            if sections["justification"]:
                st.markdown(sections["justification"], unsafe_allow_html=True)
                
                if sections["norms"]:
                    st.subheader("Normativas Aplicables")
                    for norm in sections["norms"]:
                        st.markdown(f"- {norm}")
                
                if sections["references"]:
                    st.subheader("Referencias Técnicas")
                    for ref in sections["references"][:5]:  # Mostrar solo las 5 principales
                        st.markdown(
                            f"**{ref['title']}**  \n"
                            f"*{ref['authors']} ({ref['year']})*  \n"
                            f"Fuente: {ref['source']}"
                        )
            else:
                st.warning("No se proporcionó justificación técnica")
        else:
            st.error("No se pudo generar una recomendación")
    
    with tab3:
        st.subheader("Aplicaciones Recomendadas")
        if sections["applications"]:
            st.markdown(sections["applications"])
        else:
            st.warning("No se especificaron aplicaciones para este método")

def generate_pdf_report(data: pd.Series, sections: Dict[str, str]) -> Optional[str]:
    """Genera un informe PDF profesional"""
    try:
        pdf = PDFReport()
        pdf.add_page()
        
        # Portada
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'INFORME TÉCNICO DE ESTABILIZACIÓN DE SUELOS', 0, 1, 'C')
        pdf.ln(10)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"Fecha: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
        pdf.ln(5)
        
        # Datos del suelo
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '1. Datos del Suelo', 0, 1)
        pdf.set_font('Arial', '', 11)
        
        # Construir lista de datos dinámicamente
        soil_data = [["Parámetro", "Valor"]]
        soil_data.append(["Tipo de suelo", data["Tipo de suelo"]])
        
        # Agregar solo los parámetros que tienen valores
        if 'Nivel freático (m)' in data:
            soil_data.append(["Nivel freático", f"{data['Nivel freático (m)']} m"])
        if 'Presión de carga (kPa)' in data:
            soil_data.append(["Presión de carga", f"{data['Presión de carga (kPa)']} kPa"])
        if 'Resistencia deseada (kPa)' in data and data['Resistencia deseada (kPa)'] is not None:
            soil_data.append(["Resistencia deseada", f"{data['Resistencia deseada (kPa)']} kPa"])
        
        # Granulometría
        if all(key in data for key in ['Grava (%)', 'Arena (%)', 'Limo (%)', 'Arcilla (%)']):
            soil_data.append(["Granulometría", ""])
            soil_data.append(["- Grava", f"{data['Grava (%)']}%"])
            soil_data.append(["- Arena", f"{data['Arena (%)']}%"])
            soil_data.append(["- Limo", f"{data['Limo (%)']}%"])
            soil_data.append(["- Arcilla", f"{data['Arcilla (%)']}%"])
        
        # Límites de Atterberg
        if 'Límite líquido (LL)' in data:
            soil_data.append(["Límite líquido (LL)", str(data["Límite líquido (LL)"])])
        if 'Límite plástico (LP)' in data:
            soil_data.append(["Límite plástico (LP)", str(data["Límite plástico (LP)"])])
        if 'Índice de plasticidad (IP)' in data:
            soil_data.append(["Índice de plasticidad (IP)", str(data["Índice de plasticidad (IP)"])])
        
        # Otras propiedades
        if 'Contenido de humedad (%)' in data:
            soil_data.append(["Contenido de humedad", f"{data['Contenido de humedad (%)']}%"])
        if 'pH del suelo' in data:
            soil_data.append(["pH del suelo", str(data["pH del suelo"])])
        if 'CBR (%)' in data:
            soil_data.append(["CBR", f"{data['CBR (%)']}%"])
        if 'Potencial de hinchamiento (%)' in data:
            soil_data.append(["Potencial de hinchamiento", f"{data['Potencial de hinchamiento (%)']}%"])
        
        # Imprimir tabla de datos
        col_width = pdf.w / 2.5
        for row in soil_data:
            pdf.cell(col_width, 6, row[0], border=1)
            pdf.cell(col_width, 6, row[1], border=1)
            pdf.ln()
        
        pdf.ln(10)
        
        # Evaluación de parámetros
        if sections["parameter_evaluation"]:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, '2. Evaluación de Parámetros', 0, 1)
            pdf.set_font('Arial', '', 11)
            pdf.multi_cell(0, 5, sections["parameter_evaluation"])
            pdf.ln(5)
        
        # Clasificación y problemas
        if sections["classification"] or sections["problems"]:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, '3. Análisis Técnico', 0, 1)
            
            if sections["classification"]:
                pdf.add_section("Clasificación del Suelo", sections["classification"])
            
            if sections["problems"]:
                pdf.add_section("Problemas Identificados", sections["problems"])
        
        # Recomendación y justificación
        if sections["recommendation"]:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, '4. Recomendación de Estabilización', 0, 1)
            pdf.set_font('Arial', '', 11)
            pdf.multi_cell(0, 5, sections["recommendation"])
            pdf.ln(5)
            
            if sections["justification"]:
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, '5. Justificación Técnica', 0, 1)
                pdf.set_font('Arial', '', 11)
                pdf.multi_cell(0, 5, sections["justification"])
                pdf.ln(5)
                
                if sections["norms"]:
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, 'Normativas Aplicables:', 0, 1)
                    pdf.set_font('Arial', '', 10)
                    for norm in sections["norms"]:
                        pdf.multi_cell(0, 5, f"- {norm}")
                    pdf.ln(5)
                
                if sections["references"]:
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, 'Referencias Técnicas:', 0, 1)
                    pdf.set_font('Arial', '', 10)
                    for ref in sections["references"][:5]:  # Limitar a 5 referencias
                        pdf.multi_cell(0, 5, f"- {ref['authors']} ({ref['year']}). {ref['title']}. {ref['source']}")
                    pdf.ln(5)
        
        # Aplicaciones recomendadas
        if sections["applications"]:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, '6. Aplicaciones Recomendadas', 0, 1)
            pdf.set_font('Arial', '', 11)
            pdf.multi_cell(0, 5, sections["applications"])
        
        # Guardar PDF
        temp_file = tempfile.NamedTemporaryFile(dir=TEMP_DIR, suffix=".pdf", delete=False)
        pdf_path = temp_file.name
        pdf.output(pdf_path)
        temp_file.close()
        
        return pdf_path
    
    except Exception as e:
        logger.error(f"Error generando PDF: {str(e)}")
        return None

def main():
    """Función principal del sistema"""
    try:
        st.title("Sistema Experto de Estabilización de Suelos")
        st.markdown("""
        **Herramienta profesional para recomendación de métodos de estabilización de suelos**  
        *Basado en análisis técnico, normativas internacionales y literatura científica*
        """)
        
        # Verificar API key
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Configure OPENAI_API_KEY en el archivo .env")
            st.stop()
        
        # Cargar documentos de referencia
        with st.spinner("Cargando base de conocimiento..."):
            normatives = load_documents(NORMATIVES_DIR)
            articles = load_documents(ARTICLES_DIR)
        
        # Interfaz de entrada de datos
        data, additional_info = data_input_interface()
        
        if data is not None:
            with st.spinner("Realizando análisis exhaustivo..."):
                # Generar prompt técnico
                prompt = generate_technical_prompt(
                    data.iloc[0], 
                    additional_info,
                    normatives,
                    articles
                )
                
                # Ejecutar consulta a la API
                analysis = query_ai(prompt)
                
                # Mostrar resultados
                display_results(data.iloc[0], analysis)
                
                # Opción para ver detalles completos (debug)
                if st.checkbox("Mostrar detalles completos de análisis (modo debug)"):
                    with st.expander("Respuesta completa de GPT-4"):
                        st.markdown(analysis)

                # Generar PDF
                sections = parse_response(analysis)
                pdf_path = generate_pdf_report(data.iloc[0], sections)
                
                if pdf_path:
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "Descargar Informe Completo (PDF)",
                            data=f.read(),
                            file_name="informe_estabilizacion.pdf",
                            mime="application/pdf"
                        )
                    
                    try:
                        os.unlink(pdf_path)
                    except Exception as e:
                        logger.error(f"Error al eliminar PDF temporal: {str(e)}")
    
    except Exception as e:
        st.error(f"Error en la aplicación: {str(e)}")
        logger.error(f"Error en main: {str(e)}", exc_info=True)
    
    finally:
        if os.path.exists(TEMP_DIR):
            try:
                shutil.rmtree(TEMP_DIR)
            except Exception as e:
                logger.error(f"Error al limpiar directorio temporal: {str(e)}")

if __name__ == "__main__":
    main()