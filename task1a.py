import pandas as pd
import numpy as np
import textdistance 
from itertools import product
from collections import defaultdict

#Load data from CSV file
google_small =  pd.read_csv("google_small.csv",header=0,low_memory=False)
amazon_small =  pd.read_csv("amazon_small.csv",header=0,low_memory=False)
truth_table = pd.read_csv("amazon_google_truth_small.csv",header=0,low_memory=False)

#Relacing missing values with ""
google_small.fillna("", inplace =  True)
amazon_small.fillna("", inplace =  True)

amz_id_list = []
gg_id_list = []
sim_list = []

# Function to calculate price similarity 
def price_similarity(p1,p2):
    a = abs(p1-p2)
    b = 1-(a/max(p1,p2))
    return b

THRESHOLD = 1.2
# Compare the amazon product name with google product name, if it's similar make a linkage.
for i in range(amazon_small.shape[0]):
    max_similarity = 0
    amz_title = amazon_small.loc[i]["title"]
    for j in range(google_small.shape[0]):
        gg_title = google_small.loc[j]["name"]
        gg_price = int(google_small['price'][j])
        amz_price = int(amazon_small['price'][i])
        token_1 = amz_title.split()
        token_2 = gg_title.split()
        # score is composed of 3*name+price
        price_sim = price_similarity(gg_price, amz_price)
        similarity = textdistance.jaccard(token_1, token_2)*3 + price_sim
        if similarity > max_similarity:
            max_similarity = similarity
            amz_id = amazon_small.loc[i]["idAmazon"]
            gg_id = google_small.loc[j]["idGoogleBase"]
    if max_similarity > THRESHOLD:
        sim_list.append(max_similarity)
        amz_id_list.append(amz_id)
        gg_id_list.append(gg_id)
task1a = pd.DataFrame({"idAmazon":amz_id_list, "idGoogleBase":gg_id_list})
task1a.to_csv('task1a.csv', index = False)




