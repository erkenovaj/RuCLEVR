import pickle
import psutil
import gc
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
from os import listdir
import sys

device = torch.device("cpu")
print("Using CPU.")

def save_pickle(data, file_path):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")

def get_text_data(path):
    train = pd.read_csv(train_path, encoding='utf-8')
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)
    X_spars = vectorizer.fit_transform(train.translation)
    columns = vectorizer.get_feature_names_out()
    X = pd.concat([pd.DataFrame.sparse.from_spmatrix(X_spars, columns=columns), train['image_filename']], axis=1)
    y = pd.get_dummies(train['answer'], columns=['answer'], drop_first=True, dtype='int')

    num_classes = len(y.columns)
    print(f"Found {num_classes} classes.")

    # соответствия индексов и классов
    class_mapping = dict(enumerate(y.columns))
    print("Class mapping:")
    for index, class_name in class_mapping.items():
        print(f"Index: {index}, Class: {class_name}")

    return X, y

def print_memory_usage(message=""):
    process = psutil.Process(os.getpid())
    print(f"{message} Memory usage: {process.memory_info().rss / (1024 ** 2):.2f} MB")

def get_image_data(imgs_dir, batch_size=500):
    print("Loading model...")
    model_path = ".../baselines/cnn_bow/googlenet_pretrained.pth"
    
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found.")
        return {}
    
    model = models.googlenet(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Загрузка модели на CPU
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
    batch_counter = 0
    saved_batches = 0

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
        try:
            with torch.no_grad():
                output_features = model(input_batch)
            visual_feats[img_file] = output_features.squeeze(0).numpy()
            print(f"Image {img_file} processed.")
        except Exception as e:
            print(f"Error processing image {img_file}: {e}")
            continue

        batch_counter += 1

        if batch_counter >= batch_size:
            print("Saving intermediate results...")
            intermediate_file = f"visual_feats_batch_{saved_batches}.pkl"
            save_pickle(visual_feats, intermediate_file)
            visual_feats = {} 
            batch_counter = 0
            saved_batches += 1
            gc.collect() 
            torch.cuda.empty_cache()
            print_memory_usage("After clearing cache:")

    if visual_feats:
        print("Saving final results...")
        all_visual_feats.update(visual_feats)
        final_file = "visual_feats_final.pkl"
        save_pickle(all_visual_feats, final_file)

    print("Image processing complete.")
    return saved_batches

def conc_data(X, visual_feats):
    cols = [f'vf_{i}' for i in range(len(next(iter(visual_feats.values()))))]

    visual_df = pd.DataFrame.from_dict(visual_feats, orient='index', columns=cols)
    visual_df.index.name = 'image_filename'

    print("Concatenating data...")

    X = pd.merge(X, visual_df, how='left', on='image_filename')

    print("Concatenation complete.")

    return X

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

if __name__ == '__main__':
    print("Before clearing cache:")
    print_memory_usage()
    torch.cuda.empty_cache()
    gc.collect()
    print("After clearing cache:")
    print_memory_usage()
    install('googlenet_pytorch')
    install('torchvision')

    if len(sys.argv) != 4:
        print('''
                Directory name must be specified:
                cnn_bow_inference.py <train_file_path> <val_file_path> <image directory name>
                Example: cnn_bow_inference.py ./train ./val ./images_dir
                ''')
        sys.exit(1)

    train_path = sys.argv[1]
    val_path = sys.argv[2]
    imgs_dir = sys.argv[3]

    X, y = get_text_data(train_path)
    X_val, y_val = get_text_data(val_path)

    visual_feats_path = "visual_feats_final.pkl"


    if os.path.exists("visual_feats_final.pkl"):
        all_visual_feats = {}
        print(f"Loading visual features from {visual_feats_path}...")

        final_feats = pd.read_pickle("visual_feats_final.pkl")
        all_visual_feats.update(final_feats)

    else:
        print("Extracting visual features...")
        saved_batches = get_image_data(imgs_dir)
        all_visual_feats = {}
        
        for i in range(saved_batches):
            batch_file = f"visual_feats_batch_{i}.pkl"
            if os.path.exists(batch_file):
                batch_feats = pd.read_pickle(batch_file)
                all_visual_feats.update(batch_feats)
        
        if os.path.exists("visual_feats_final.pkl"):
            final_feats = pd.read_pickle("visual_feats_final.pkl")
            all_visual_feats.update(final_feats)

    X = conc_data(X, all_visual_feats)
    X_val = conc_data(X_val, all_visual_feats)
    
    mlp = MLPClassifier(random_state=1, max_iter=300, activation='relu', solver='adam').fit(X.drop('image_filename', axis = 1), y)
    X_val = X_val.drop('image_filename', axis = 1)
    y_pred = mlp.predict(X_val)
    print(classification_report(y_val, y_pred, zero_division=0))
