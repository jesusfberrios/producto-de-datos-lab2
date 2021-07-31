from fastapi import FastAPI, HTTPException
import io
import numpy as np
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

# Asignamos una instancia de la clase FastAPI a la variable "app".
# Interacturaremos con la API usando este elemento.
app = FastAPI(title='Implementando un modelo de Machine Learning usando FastAPI')

# Enlistamos los modelos disponibles usando Enum. Útil cuando tenemos opciones predefinidas.
class Model(str, Enum):
    yolov3tiny = "yolov3-tiny"
    yolov3 = "yolov3"


# Usando @app.get("/") definimos un método GET para el endpoint / (que sería como el "home").
@app.get("/")
def home():
    return "¡Felicitaciones! Tu API está funcionando según lo esperado. Anda ahora a https://producto-datos-lab2.herokuapp.com/docs."


# Este endpoint maneja la lógica necesaria para detectar objetos.
# Requiere como entrada el modelo deseado y la imagen.
@app.post("/predict") 
def prediction(model: Model, file: UploadFile = File(...)):

    # 1. Validar el archivo de entrada
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Tipo de archivo no soportado.")
    
    # 2. Transformar la imagen cruda a una imagen CV2
    
    # Leer la imagen como un stream de bytes
    image_stream = io.BytesIO(file.file.read())
    
    # Empezar el stream desde el principio (posicion cero)
    image_stream.seek(0)
    
    # Escribir el stream en un numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    
    # Decodificar el numpy array como una imagen
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    
    # 3. Correr el modelo de detección de objetos
    
    # Correr la detección de objetos
    bbox, label, conf = cv.detect_common_objects(image, model=model)
    
    # Crear una imagen que contenga las cajas delimitadoras y etiquetas
    output_image = draw_bbox(image, bbox, label, conf)
    
    # Guardarla en un directorio del server
    cv2.imwrite(f'/tmp/{filename}', output_image)
    
    
    # 4. Transmitir la respuesta de vuelta al cliente
    
    # Abrir la imagen para leerla en formato binario
    file_image = open(f'/tmp/{filename}', mode="rb")
    
    # Retornar la imagen como un stream usando un formato específico
    return StreamingResponse(file_image, media_type="image/jpeg")
