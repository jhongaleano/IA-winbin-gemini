import json
import logging
import os
import io
import httpx
import base64
import re
from fastapi.middleware.cors import CORSMiddleware

from groq import Groq, APIError
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
        response = cloudinary.uploader.upload(file_content, folder= "winbin_detecciones")
        return response['secure_url']
    except Exception as e:
        logging.error(f"Error al subir a cloudinary: {e}")
        return None

app = FastAPI(
    title="Servicio de Detección de Reciclaje con GROQ - WinBin",
    description="API en Python para procesar imágenes usando Llama 3.2 Vision.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health():
    return {"status": "ok", "message": "Servicio de IA activo (Groq)"}

@app.post("/api/ia-analisis")
async def analizarImagen(
    id_session: str = Form(...), 
    file: UploadFile = File(...),
    authorization: str = Header(...)
    ):
    contents =  await file.read()
    
    if not contents:
        raise HTTPException(status_code=400, detail="Archivo vacío.")

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de autorización inválido o ausente"
        )
   
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo debe ser una imagen (jpeg, png, webp, etc.).",
        )

    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()  
        file.file.seek(0)  
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se pudo leer la imagen. Archivo corrupto o formato no soportado.",
        )
        
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Configuración de Groq incompleta. Falta GROQ_API_KEY.",
        )

    # 1. Convertir la imagen a Base64 para Groq
    base64_image = base64.b64encode(contents).decode('utf-8')
    mime_type = file.content_type

    prompt = """
    Analiza la imagen adjunta y determina si contiene cartón o botellas de plástico. Sigue estrictamente esta estructura en tu respuesta:

1. Detección de Objetos:
- Identifica si hay "Cartón", "Botella de plástico" o "Ninguno".

2. Estimación de Tamaño:
- Clasifica el tamaño de cada objeto detectado en una de las siguientes categorías:
- pequeno (ej. botellas de 275 ml o menos, pedazos pequeños de cartón menores a 20x20 cm)
- mediano (ej. botellas de 500 ml a 1 litro, cajas de zapatos o empaques medianos)
- grande (ej. botellas de 1.5 litros a 3 litros, garrafones de agua, cajas grandes de mudanza o empaques voluminosos)

Si no detectas cartón ni botellas de plástico:
- status: "no_reciclable"
- confiabilidad_porcentaje: 0.0
- material: null
- tamano: null

Si detectas un objeto reciclable válido:
- status: "exito"
- material: "Botella" o "Carton"
- tamano: "pequeno", "mediano" o "grande"
- confiabilidad_porcentaje: (un número entre 0 y 100)

CRÍTICO: Debes devolver ÚNICAMENTE un objeto JSON válido con las claves mencionadas. No incluyas texto antes ni después del JSON. No uses bloques de código Markdown (```json).
    """
    
    try: 
        # 2. Inicializar cliente Groq y hacer la petición multimodal
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b", # Modelo de visión actual en Groq
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=256,
        )
        
        response_text = response.choices[0].message.content
        
    except APIError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error en la API de Groq: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error inesperado al procesar la imagen: {str(e)}",
        )      

    if not response_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="El modelo no devolvió un análisis válido.",
        )
        
    # 3. Limpiar posibles bloques Markdown del JSON devuelto por Llama
    clean_json = re.sub(r'^```json\s*', '', response_text)
    clean_json = re.sub(r'\s*```$', '', clean_json).strip()

    try:
        analysis_data = RecyclableItem.model_validate_json(clean_json)
    except ValidationError as e:
        logging.error(f"Error de validación JSON. Respuesta cruda: {response_text}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="La respuesta del modelo no cumple el formato esperado.",
        )

    if analysis_data.status == "no_reciclable" or analysis_data.confiabilidad_porcentaje < 60 or not analysis_data.material or not analysis_data.tamano:
        return {
            "status":"no_reciclable",
            "objeto_detectado":"Desconocido/No valido",
            "acertacion_confianza":analysis_data.confiabilidad_porcentaje,
            "msg": "El objeto no coincide con botellas de plastico o carton con suficiente confianza"
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
        "utlImagen": url_publica,
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
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            peticion = await http_client.post(url_api_java, json=datos_java, headers=headers)
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
        "status": analysis_data.status,
        "objeto_detectado": analysis_data.material,
        "acertacion_confianza": analysis_data.confiabilidad_porcentaje,
        "tamano_calculado": analysis_data.tamano,
        "id_categoria_puntaje": id_categoria,  
        "envio_backend": respuesta_java,
        "id_material": id_material
    }