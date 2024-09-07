from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
import uuid
import DetCls  # Import your DetCls module
import os

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_image(request: Request, file: UploadFile = File(...)):
    # Define paths
    temp_file_location = f"temp_{uuid.uuid4()}_{file.filename}"
    static_file_location = f"static/images/{uuid.uuid4()}_{file.filename}"
    
    # Save the uploaded file temporarily
    with open(temp_file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Call detectclassify function
    annotated_image_path, predicted_classes = DetCls.detectClassify(temp_file_location)

    # Ensure the annotated image exists
    if not os.path.exists(annotated_image_path):
        return {"error": "Annotated image not found"}

    # Copy annotated image to static/images
    annotated_image_name = os.path.basename(annotated_image_path)
    shutil.copy(annotated_image_path, static_file_location)

    # Prepare the URL for the annotated image
    annotated_image_url = static_file_location.replace("static/", "/static/")
    print(annotated_image_url)
    # Remove temporary files
    os.remove(temp_file_location)
    os.remove(annotated_image_path)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "annotated_image_url": annotated_image_url,
        "predicted_classes": predicted_classes,
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
