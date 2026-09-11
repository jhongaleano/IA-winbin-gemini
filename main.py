import json
import logging
import os
import io
import httpx
from fastapi.middleware.cors import CORSMiddleware

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Form, Header


import cloudinary
import cloudinary.uploader

load_dotenv()

cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME") ,
    api_key = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET")
)

SpringBoot_url = os.getenv("SPRINGBOOT_URL")

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
    status: Literal["exito", "no_reciclable"]
    confiabilidad_porcentaje: float = Field(ge=0, le=100)
    material: Optional[Literal["Botella", "Carton"]] = None
    tamano: Optional[Literal["pequeno", "mediano", "grande"]] = None



def subir_cloudinary(file_content, filename):
    if not all([os.getenv("CLOUDINARY_CLOUD_NAME"), os.getenv("CLOUDINARY_API_KEY"), os.getenv("CLOUDINARY_API_SECRET")]):
        raise HTTPException(status_code=500, detail="Configuración de Cloudinary incompleta.")

    try:
        response = cloudinary.uploader.upload(file_content, public_id=filename)
        return response['secure_url']
    except Exception as e:
        logging(f"Error al subiar a cloudinary: {e}")
        return None

app = FastAPI(
    title="Servicio de Detección de Reciclaje con GEMINI - WinBin",
    description="API en Python para procesar imágenes.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Permite peticiones desde cualquier origen (Flutter, Web, etc.)
    allow_credentials=True,
    allow_methods=["*"],            # Permite todos los métodos HTTP (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],            # Permite todos los encabezados (Authorization, Content-Type, etc.)
)

@app.post("/api/ia-analisis")
async def analizarImagen(
    id_session: str = Form(...), 
    file: UploadFile = File(...),
    authorization: str = Header(...)
    ):
    contents =  await file.read()
    

    if not contents:
        raise HTTPException(status_code=400, detail="archivo vacio. ")

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
        
        image = Image.open(io.BytesIO(contents))
        image.verify()  # detecta corrupción básica
        file.file.seek(0)  # verify() deja el puntero al final
        image = Image.open(io.BytesIO(contents))

        
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
    Analiza la imagen adjunta y determina si contiene cartón o botellas de plástico. Sigue estrictamente esta estructura en tu respuesta:

1. Detección de Objetos:
   - Identifica si hay "Cartón", "Botella de plástico" o "Ninguno".


2. Estimación de Tamaño:
   - Clasifica el tamaño de cada objeto detectado en una de las siguientes categorías:
     * Pequeño (ej. botellas de 275 ml o menos, pedazos pequeños de cartón menores a 20x20 cm)
     * Mediano (ej. botellas de 500 ml a 1 litro, cajas de zapatos o empaques medianos)
     * Grande (ej.  botellas de 1.5 litros a 3 litros, garrafones de agua, cajas grandes de mudanza o empaques voluminosos)
   - Proporciona una dimensión aproximada estimada (en centímetros o litros) basándote en objetos de referencia visibles en la imagen.

Si no detectas cartón ni botellas de plástico:
- status: "no_reciclable"
- confiabilidad_porcentaje: 0.0
- material: null
- tamano: null

Si detectas un objeto reciclable válido:
- status: "exito"
- material, tamano y confiabilidad_porcentaje obligatorios
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


    if analysis_data.status == "no_reciclable" or analysis_data.confiabilidad_porcentaje < 60 or not analysis_data.material or not analysis_data.tamano:
        return {
            "status":"no_reciclable",
            "objeto_detectato":"Desconocido/No valido",
            "acertacion_de_confianza":analysis_data.confiabilidad_porcentaje,
            "msg": "El obejto no coincide con botellas de plastico o carton con suficiente confianza"
        }
    


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
    

    url_publica = subir_cloudinary(contents,file.filename)

    if not url_publica:
        raise HTTPException(status_code=400,detail="No se pudo subir la imagen a clodinary.")

    datos_java = {
        "confianza": analysis_data.confiabilidad_porcentaje,
        "utlImagen": url_publica, # URL de la imagen en la nube
        "idSession": id_session,
        "idMaterial": id_material,
        "idCategoria": id_categoria
    }

    headers = {
        "Authorization": authorization,
        "Content-Type": "application/json"
    }

    url_api_java = f"{SpringBoot_url}api/registroia/guardar-resultado"
    respuesta_java = {"status_code_java": 500, "info": "Error desconocido"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            peticion = await client.post(url_api_java, json=datos_java, headers=headers)
            if peticion.status_code in [200, 201]:
                respuesta_java = {
                    "status_code_java": 200,
                    "msg": "Resultado enviado a Spring Boot con éxito."
                }
            else:
                respuesta_java = {
                    "status_code_java": peticion.status_code,
                    "msg": "Error al enviar a Spring Boot: " + peticion.text,
                }
    except Exception as e:
        respuesta_java = {
            "status_code_java": 500,
            "msg": "Java backend no alcanzable: " + str(e),
        }

    return {
        "status": "exito",
        "objeto_detectado": analysis_data.material,
        "acertacion_confianza": analysis_data.confiabilidad_porcentaje,
        "tamano_calculado": analysis_data.tamano,
        "id_categoria_puntaje": id_categoria,  
        "envio_backend": respuesta_java,
        "id_material": id_material
    }


