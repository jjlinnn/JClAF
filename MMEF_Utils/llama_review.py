import pandas as pd
import ollama
import csv
import os

def get_summary(description):
    response = ollama.chat(model='llama3.2:1b', messages=[
        {
            'role': 'user',
            'content': f"Only give me a sentence based on the following collected user reviews about the product, be careful that you mustn't give me several rows of sentences, only a whole sentence. Generate a single, comprehensive sentence that effectively summarizes the product's key features and customer opinions on this product: {description}, make sure the output is under 100 words.",
        },
    ])
    return response['message']['content']

input_file = 'aggregated_reviews_sports.csv'
output_file = 'output_sports.csv'

df = pd.read_csv(input_file)

if os.path.exists(output_file):
    processed_asins = set(pd.read_csv(output_file)['asin'])  
else:
    processed_asins = set()

with open(output_file, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    if not processed_asins:
        writer.writerow(['asin', 'summary'])
    
    for index, row in df.iterrows():
        asin = row['itemID']
        review = row['reviewText']
        
        if asin in processed_asins:
            print(f"Skipping already processed ASIN: {asin}")
            continue
        
        try:
            summary = get_summary(review)
        except Exception as e:
            print(f"Error processing ASIN {asin}: {e}")
            continue
        
        writer.writerow([asin, summary])
        
        processed_asins.add(asin)
        
        print(f"Processed row {index + 1}/{len(df)}: {asin}")

print("Summary generation completed and saved to output_sports.csv")
