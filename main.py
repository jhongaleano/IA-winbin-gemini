import json
import os
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Form, Header



load_dotenv()

MATERIALES_MAP = {
    "Botella": 1,
    "Carton": 2
}

CATEGORIAS_MAP = {
    (1,"pequeno"): 1,
    (1,"mediano"): 2,
    (1,"grande"): 3,
    (2,"pequeno"): 4,
    (2,"mediano"): 5,
    (2,"grande"): 6,
}

class RecyclableItem(BaseModel):
    material: Literal["Botella","Carton"] = Field(
        description="Material identificado en la imagen."
    )
    confiabilidad_porcentaje: float = Field(
        description="Porcentaje de certeza de la identificación (0.0 a 100.0)."
    )
    tamano: Literal["pequeno", "mediano", "grande"] = Field(
        description="Estimación del tamaño del objeto."
    )



app = FastAPI(
    title="Servicio de Detección de Reciclaje con YOLO - WinBin",
    description="API en Python para procesar imágenes con dos modelos especializados y retornar id_categoria.",
    version="1.1.0"
)

@app.post("/api/ia-analisis")
async def analizarImagen(
    id_session: int = Form(...), 
    file: UploadFile = File(...),
    authorization: str = Header(...)
    ):

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de autorizacion invalido o ausente"
        )
   

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo debe ser una imagen (jpeg, png, webp, etc.).",
        )

    try:
        image = Image.open(file.file)
        image.verify()  # detecta corrupción básica
        file.file.seek(0)  # verify() deja el puntero al final
        image = Image.open(file.file)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se pudo leer la imagen. Archivo corrupto o formato no soportado.",
        )
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Configuración del servicio incompleta.",
        )

    prompt = """
    Analiza la imagen adjunta e identifica si hay cartón o botella de plastico.
    Reglas para asignación de puntos:
    - Botella de plástico/vidrio: Pequeña (10 pts), Mediana (20 pts), Grande (30 pts).
    - Cartón: Pequeño (15 pts), Mediano (25 pts), Grande (40 pts).
    - Otro material: 0 pts.

    Devuelve la información estructurada respetando el esquema.
    """
    try: 
        client = genai.Client()
        response = client.models.generate_content(
            model="models/gemini-3.6-flash",
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", # Fuerza respuesta en formato JSON
                response_schema=RecyclableItem,        # Aplica el esquema Pydantic
                temperature=0.1,                       # Baja temperatura para mayor precisión
            ),
        )
    except genai_errors.ClientError as e:
        # Errores 4xx de la API (cuota, permisos, etc.)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error al consultar el modelo de IA: {e}",
        )
    except genai_errors.ServerError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El servicio de IA no está disponible temporalmente.",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error inesperado al procesar la imagen.",
        )      

    if not response.text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="El modelo no devolvió un análisis válido.",
        )
    try:
        analysis_data = RecyclableItem.model_validate_json(response.text)
    except ValidationError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="La respuesta del modelo no cumple el formato esperado.",
        )
    


    id_material = MATERIALES_MAP.get(analysis_data.material)

    id_categoria = CATEGORIAS_MAP.get((id_material, analysis_data.tamano))

    if id_material is None or id_categoria is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "mensaje": "Material o tamaño no reconocido en el sistema.",
                "material": analysis_data.material,
                "tamano": analysis_data.tamano,
            },
        )
    if analysis_data.confiabilidad_porcentaje < 60:
     raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={
            "mensaje": "Confianza insuficiente para clasificar el objeto.",
            "confianza": analysis_data.confiabilidad_porcentaje,
        },
    )
    return {
        "confianza": analysis_data.confiabilidad_porcentaje,
        "utl_imagen": image, # URL de la imagen en la nube
        "id_session": id_session,
        "id_categoria": id_categoria,
        "id_material": id_material
    }


