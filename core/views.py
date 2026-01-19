from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
from .ml.emotion_detector import detect_emotion as ml_detect_emotion
from .ml.tumor_detector import detect_tumor as ml_detect_tumor

def index(request):
    return render(request, 'index.html')

def predict_emotion(request):
    if request.method == 'POST' and request.FILES['image']:
        myfile = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        
        # Absolute path for processing
        file_path = os.path.join(settings.MEDIA_ROOT, filename)
        
        # Process image
        processed_path, label = ml_detect_emotion(file_path)
        
        # Get URL for processed image
        if processed_path:
            processed_filename = os.path.basename(processed_path)
            processed_url = fs.url(processed_filename)
        else:
            processed_url = uploaded_file_url
            label = "Error en procesamiento"

        return render(request, 'result.html', {
            'original_url': uploaded_file_url,
            'processed_url': processed_url,
            'label': label,
            'type': 'Emoción'
        })
    return render(request, 'index.html')

def predict_tumor(request):
    if request.method == 'POST' and request.FILES['image']:
        myfile = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        
        # Absolute path
        file_path = os.path.join(settings.MEDIA_ROOT, filename)
        
        # Process
        processed_path, label = ml_detect_tumor(file_path)
        
        if processed_path:
            processed_filename = os.path.basename(processed_path)
            processed_url = fs.url(processed_filename)
        else:
            processed_url = uploaded_file_url
            label = "Error"
            
        return render(request, 'result.html', {
            'original_url': uploaded_file_url,
            'processed_url': processed_url,
            'label': label,
            'type': 'Tumor Cerebral'
        })
    return render(request, 'index.html')
