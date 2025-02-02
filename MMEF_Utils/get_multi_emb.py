import requests
from transformers import BlipProcessor, BlipModel
from PIL import Image
import torch
import pandas as pd
from io import BytesIO
import pickle
import os
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

df = pd.read_csv('img_url_sports.csv')

df['text'] = df['description'] + ' ' + df['summary']

def generate_noise_image(size=(224, 224), color_mode='L'):
    if color_mode == 'RGB':
        noise = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    elif color_mode == 'L':
        noise = np.random.randint(0, 256, (size[1], size[0]), dtype=np.uint8)
    else:
        raise ValueError("Unsupported color mode. Use 'RGB' or 'L'.")
    
    return Image.fromarray(noise, color_mode)

def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)  
        response.raise_for_status() 
        img = Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error loading image from URL {url}: {e}. Using noise image as placeholder.")
        img = generate_noise_image()  
    return img

def get_multimodal_embedding(text, img):
    text = str(text)
    inputs = processor(images=img, text=text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}  
    outputs = model(**inputs)
    text_embedding = outputs.text_embeds  # [batch_size, hidden_size]
    image_embedding = outputs.image_embeds
    multimodal_embedding = torch.cat([text_embedding, image_embedding], dim=1)
    return multimodal_embedding

embeddings = {}

all_embeddings = []

save_path = 'multimodal_embeddings_sports.pkl'

if os.path.exists(save_path):
    with open(save_path, 'rb') as f:
        embeddings = pickle.load(f)

with torch.no_grad():
    for index, row in df.iterrows():
        asin = row['itemID']
        text = row['text']
        img_url = row['imUrl']

        if asin in embeddings:
            print(f"Item {asin} already processed. Skipping.")
            continue

        if pd.isnull(text) and pd.isnull(img_url):
            if len(all_embeddings) > 0:
                mean_embedding = torch.mean(torch.stack(all_embeddings), dim=0)
                embeddings[asin] = mean_embedding
            else:
               embeddings[asin] = torch.zeros(1, 512 * 2, device=device)   
        else:
            if pd.notnull(img_url):
                img = load_image_from_url(img_url)
                embedding = get_multimodal_embedding(text, img)
            elif pd.notnull(text):
                img = Image.new('RGB', (224, 224), color='white')  
                embedding = get_multimodal_embedding(text, img)
            
            embeddings[asin] = embedding
            all_embeddings.append(embedding)

            print(f"Processed ASIN: {asin}")

        if (index + 1) % 50 == 0:
            print(f"Saving embeddings after processing {index + 1} items...")
            with open(save_path, 'wb') as f:
                pickle.dump(embeddings, f)

with open(save_path, 'wb') as f:
    pickle.dump(embeddings, f)

print("Finished processing and saved embeddings.")
