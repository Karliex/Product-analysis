import pandas as pd

# read all datasets
amazon = pd.read_csv("amazon.csv")
google = pd.read_csv("google.csv")

google_copy = google.copy()
amazon_copy = amazon.copy()


#data preprocessing
# a function changes British pound to Australian dollar
def toAUD(price):
    price = round(float(price.split()[0])*1.87,2)
    return price

# unify the type of the price
for i in range(google.shape[0]):
    try:
        float(google_copy.loc[i]["price"])
    except:
        google_copy.loc[i]["price"] = toAUD(google_copy.loc[i]["price"])

for i in range(google.shape[0]):
    try: 
        google_copy["price"][i] = float(google_copy.loc[i]["price"])
    except:
        google_copy["price"][i] = np.nan

# cutting bins for different level of prices (low, medium, high, ultra high)
bin_list = []

for num in range(0, 500, 20):
    bin_list.append(num)
for num in range(500, 1000, 100):
    bin_list.append(num)
for num in range(1000, 10000, 1000):
    bin_list.append(num)
for num in range(10000, 60000, 5000):
    bin_list.append(num)


# allocate prices into responding blocks
google_copy['label'] = pd.cut(google_copy['price'], bin_list, right = False)
amazon_copy['label'] = pd.cut(amazon_copy['price'], bin_list, right = False)

amazon_b = pd.DataFrame({"block_key":amazon_copy['label'],"product_id":amazon["idAmazon"]})
amazon_b.to_csv('amazon_blocks.csv', index = False, encoding = 'utf-8')

google_b = pd.DataFrame({"block_key":google_copy['label'],"product_id":google["id"]})
google_b.to_csv('google_blocks.csv', index = False, encoding = 'utf-8')





