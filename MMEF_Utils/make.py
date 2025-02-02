import pandas as pd
import json
file_pathre = "reviews_Beauty.json"


with open(file_pathre, 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

dfre = pd.DataFrame(data)#reviewerID": "A28O3NP6WR5517", "asin
print(dfre.head())
file_pathin = "beauty-indexed-v4.inter"


df = pd.read_csv(file_pathin, sep='\t')
print(df.head())

filtered_df = df[df['x_label'] == 0]
mapping_dfu = pd.read_csv("u_id_mapping_beauty.csv", sep='\t')
user_mapping = dict(zip(mapping_dfu['userID'], mapping_dfu['user_id']))#user_id	userID
mapping_dfi = pd.read_csv("i_id_mapping_beauty.csv", sep='\t')
item_mapping = dict(zip(mapping_dfi['itemID'], mapping_dfi['asin']))#asin	itemID
filtered_df['itemID'] = filtered_df['itemID'].map(item_mapping)
filtered_df['userID'] = filtered_df['userID'].map(user_mapping)

print(filtered_df)

filtered_df.to_csv("train_beauty.csv", sep='\t')
dfre.rename(columns={'reviewerID': 'userID', 'asin': 'itemID'}, inplace=True)
print(dfre)
filtered_dfre = dfre.merge(filtered_df[['userID', 'itemID']], on=['userID', 'itemID'])
print(filtered_dfre.head())
filtered_dfre.to_csv("trainre_beauty.csv", sep='\t')

aggregated_df = filtered_dfre.groupby('itemID', as_index=False).agg({'reviewText': ' '.join})
print(aggregated_df)

aggregated_df.to_csv("aggregated_reviews_beauty.csv", index=False)