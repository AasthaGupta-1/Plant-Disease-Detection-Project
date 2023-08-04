from fastapi import FastAPI, File, UploadFile , Request ,Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
app = FastAPI()

# origins = [
#     "http://localhost",
#     "http://localhost:5500",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

templates=Jinja2Templates(directory="C:/Users/hp/Documents/Project/tomato-disease/training/apis/website")
MODEL = tf.keras.models.load_model("C:/Users/hp/Documents/Project/tomato-disease/training/full_models/1")

CLASS_NAMES = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

predicted_class="None"
confidence=1.0
@app.get("/home/{user_name}")
async def ping(request: Request):
    return templates.TemplateResponse("Plant_Disease_Detection.html",{"request":request })

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    request: Request,file: UploadFile = File(...)
):
    print(file.filename)
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    # ping(predicted_class,float(confidence))
    result = f"Class: {predicted_class},"+"\n"+f"Confidence: {float(confidence)}"
    return templates.TemplateResponse("Plant_Disease_Detection.html", {"request": request, "result": result})
    # return {
    #     'class': predicted_class,
    #     'confidence': float(confidence)
    # }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
