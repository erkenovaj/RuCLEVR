import torchvision.models as models
from PIL import Image
import os
from os.path import join
import pip
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
import numpy as np
import random

from googlenet_pytorch import GoogLeNet
import torch
import torchvision.transforms as transforms
from PIL import Image
from os import listdir
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cpu":
    print("CUDA is not available. Using CPU.")

def get_text_data(path):
    train = pd.read_csv(train_path, encoding='utf-8')
    vectorizer = TfidfVectorizer(max_df = 0.9, min_df = 2)
    X_spars = vectorizer.fit_transform(train.translation)
    columns=vectorizer.get_feature_names_out()
    X = pd.DataFrame.sparse.from_spmatrix(X_spars, columns = columns)
    X['image_filename'] = train['image_filename']
    y = pd.get_dummies(train['answer'], columns=['answer'], drop_first= True, dtype='int')
    return X, y


def get_image_data(imgs_dir):
    print("Loading model...") 

    model = GoogLeNet.from_pretrained("googlenet")
    model.eval() 
    
    print("Model loaded successfully.")  
    print(model)  

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    print("Listing image files...")
    files = listdir(imgs_dir)
    print(f"Found {len(files)} image files.") 
    visual_feats = {}
    for img_file in files:
        print(f"Processing image: {img_file}") 
        full_img_path = join(imgs_dir, img_file) 
        
        try:
            input_image = Image.open(full_img_path)
        except Exception as e:
            print(f"Error opening image {img_file}: {e}")
            continue
        
        if input_image.mode == 'RGBA':
            input_image = input_image.convert('RGB')
        
        input_tensor = preprocess(input_image)
        if input_tensor.shape[0] != 3:
            print(f"Ignoring {img_file}: unexpected number of channels")
            continue
        input_batch = input_tensor.unsqueeze(0)
        output_features = model(input_batch)
        visual_feats[img_file] = output_features.squeeze(0).detach().numpy()
        print(f"Image {img_file} processed.") 
    print("Image processing complete.")
    return visual_feats

def conc_data(X, visual_feats):
    cols = [f'vf_{i}' for i in range(len(torch.flatten(torch.tensor(random.choice(list(visual_feats.values()))))))]
    for index, row in X.iterrows():
        if row['image_filename'] in visual_feats:
            X[cols] = [float(num) for num in torch.flatten(torch.tensor(visual_feats[row['image_filename']]))]
    return X

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

if __name__ == '__main__':
    install('googlenet_pytorch')
    install('torchvision')

    if len(sys.argv) != 4:
        print('''
                Directory name must be specified:
                cnn_bow.py <train_file_path> <val_file_path> <image directory name>
                Example: ram_area_statistics.py ./test
                ''')
        sys.exit(1)

    train_path = sys.argv[1]
    val_path = sys.argv[2]
    imgs_dir = sys.argv[3]

    train_path = r'CLEVR_\train_questions.csv'
    val_path = r'CLEVR_\val_questions.csv'
    X, y = get_text_data(train_path)
    X_val, y_val = get_text_data(val_path)

    visual_feats = get_image_data(imgs_dir)
    X = conc_data(X, visual_feats)
    X_val = conc_data(X_val, visual_feats)

    mlp = MLPClassifier(random_state=1, max_iter=300, activation='relu', solver='adam').fit(X.drop('image_filename', axis = 1), y)
    X_val = X_val.drop('image_filename', axis = 1)
    y_pred = mlp.predict(X_val)
    print(classification_report(y_val, y_pred, zero_division=0))