import uvicorn
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Dict
import base64
import numpy as np
import pandas as pd
import data_processing
import model_tester  
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse
import time

app = FastAPI()

project_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_directory)

model_paths = [
    {
        'name': 'SVM_HARUS',
        'path': 'models/SVM_HARUSOLD.pkl',
        'file_type': 'csv'
    },
    {
        'name': 'Neural_Network_Accelerometer_Raw',
        'path': 'models/Neural_Network_Accelerometer_Raw.ckpt',
        'file_type': 'txt'
    },
    {
        'name': 'U_net_raw',
        'path': 'models/U-net.h5',
        'file_type': 'csv'
    },
]

@app.get("/")
async def get():
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    return FileResponse(index_path)

@app.post("/api/recognize")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        file_ext = file.filename.split('.')[-1]
        file_type = 'csv' if file_ext.lower() == 'csv' else 'txt'

        contents = await file.read()

        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Файл пустой")

        try:
            file.file.seek(0)
            if file_type == 'csv':
                data = pd.read_csv(file.file)
            elif file_type == 'txt':
                data = np.loadtxt(file.file)
                data = pd.DataFrame(data) 
            else:
                raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла")

            print(data.head())  

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при чтении файла: {str(e)}")

        X_test, _ = data_processing.process_data(data)
        features = data_processing.generate_features(X_test)

        # Построение графика
        fig, ax = plt.subplots()
        ax.plot(data.iloc[:, 0])  # Предположим, что первый столбец - это время
        ax.set_title(f"График значений")
        fig.savefig('output_image.png')
        plt.close(fig)

        with open('output_image.png', 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        results = []
        
        tester = model_tester.ModelTester()

        for model_info in model_paths:
            model_path = model_info['path']
            start_time = time.time()
            if model_path.endswith('.pkl'):
                model_obj = tester.load_svm_model(model_path)
                report = tester.test_model(model_obj, X_test, _)
                end_time = time.time()
                result = {
                    "method": model_info['name'],
                    "class": report,
                    "time": end_time - start_time
                }
            elif model_path.endswith('.h5'):
                model_obj = tester.load_keras_model(model_path)
                report = tester.test_keras_model(model_obj, features, _)
                end_time = time.time()
                result = {
                    "method": model_info['name'],
                    "class": report,
                    "time": end_time - start_time
                }
            else:
                sess = tester.load_tf_model(model_path)
                report = tester.test_tf_model(sess, features, features, 'Placeholder:0', 'add_3:0')
                end_time = time.time()
                result = {
                    "method": model_info['name'],
                    "class": report,
                    "time": end_time - start_time
                }
            results.append(result)

        return JSONResponse(content={"image": encoded_image, "results": results})

    except Exception as e:
        print(f"Ошибка обработки запроса: {str(e)}")
        current_directory = os.getcwd()
        print(f"Текущая директория: {current_directory}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8088, reload=True)