#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("zomato.csv")
print(df.columns.tolist())
print(df.shape)
df


# In[2]:


df.isnull().sum()


# In[3]:


import re

def get_avg_rating(review):
    ratings = re.findall(r'Rated (\d\.\d)', str(review))
    ratings = [float(r) for r in ratings]
    if len(ratings) > 0:
        return sum(ratings) / len(ratings)
    else:
        return None

sample = df[df['rate'].isnull()][['reviews_list']].head(5).copy()

sample['extracted_rating'] = sample['reviews_list'].apply(get_avg_rating)

print(sample)


# In[4]:


df['rate'] = df.apply(
    lambda row: get_avg_rating(row['reviews_list']) if pd.isnull(row['rate']) else row['rate'],
    axis=1
)


# In[5]:


d1=df.copy()
d1=df.dropna(subset=['rate'])
print(d1.shape)
d1.isnull().sum()


# In[6]:


sample_before = d1[['cuisines']].head(5).copy()

d2 = d1.copy()
d2['cuisines'] = d2['cuisines'].str.split(',').str[0]
d2['cuisines'] = d2['cuisines'].fillna(d2['cuisines'].mode()[0])

sample_after = d2[['cuisines']].head(5)

comparison = pd.DataFrame({
    'Before': sample_before['cuisines'],
    'After': sample_after['cuisines']
})

print(comparison)


# In[7]:


d3 = d2.copy()

d3['approx_cost(for two people)'] = d3['approx_cost(for two people)'].str.replace(',', '')


d3['approx_cost(for two people)'] = pd.to_numeric(
    d3['approx_cost(for two people)'], errors='coerce'
)

median_value = d3['approx_cost(for two people)'].median()

d3['approx_cost(for two people)'] = d3['approx_cost(for two people)'].fillna(median_value)
print("\nMedian Value Used:", median_value)
print(d3.isnull().sum())


# In[8]:


d3['rest_type'] = d3['rest_type'].str.split(',').str[0]
mode_value = d3['rest_type'].mode()[0]
d3['rest_type']=d3['rest_type'].fillna(mode_value)
print("Mode Value Used:", mode_value)
print(d3.isnull().sum())


# In[9]:


cols_to_drop = ['phone', 'menu_item', 'url', 'address', 'dish_liked']

d3.drop(columns=cols_to_drop, inplace=True)

print(d3.columns)


# In[10]:


print(d3.isnull().sum())


# In[11]:


d3['rate'] = d3['rate'].str.replace('/5', '')
d3['rate'] = pd.to_numeric(d3['rate'], errors='coerce')
d3['approx_cost(for two people)'] = d3['approx_cost(for two people)'].astype(str).str.replace(',', '')
d3['approx_cost(for two people)'] = pd.to_numeric(d3['approx_cost(for two people)'], errors='coerce')


# In[12]:


d3_clean = d3.copy()

cols = ['votes', 'approx_cost(for two people)','rate']

for col in cols:
    total_rows = len(d3_clean)

    Q1 = d3_clean[col].quantile(0.25)
    Q3 = d3_clean[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = d3_clean[(d3_clean[col] < lower) | (d3_clean[col] > upper)]
    count = len(outliers)
    percentage = (count / total_rows) * 100

    print(f"\n{col}: {count} outliers ({percentage:.2f}%)")

    if percentage < 5:
        print("→ Removing outliers")
        d3_clean = d3_clean[(d3_clean[col] >= lower) & (d3_clean[col] <= upper)]
    else:
        print("→ Capping outliers")
        d3_clean[col] = d3_clean[col].clip(lower, upper)


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.boxplot(y=d3['votes'])
plt.title("Votes (Before Outlier Removal)")

plt.subplot(1,2,2)
sns.boxplot(y=d3_clean['votes'])
plt.title("Votes (After Outlier Removal)")

plt.tight_layout()
plt.show()


# In[14]:


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.boxplot(y=d3['approx_cost(for two people)'])
plt.title("Cost (Before Outlier Removal)")

plt.subplot(1,2,2)
sns.boxplot(y=d3_clean['approx_cost(for two people)'])
plt.title("Cost (After Outlier Removal)")

plt.tight_layout()
plt.show()


# In[15]:


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.boxplot(y=d3['rate'])
plt.title("Rate (Before Outlier Removal)")

plt.subplot(1,2,2)
sns.boxplot(y=d3_clean['rate'])
plt.title("Rate (After Outlier Removal)")

plt.tight_layout()
plt.show()


# In[16]:


d3[['votes', 'approx_cost(for two people)', 'rate']].skew()


# In[17]:


d3[['votes', 'approx_cost(for two people)', 'rate']].kurt()


# In[20]:


d3_clean[['votes', 'approx_cost(for two people)', 'rate']].skew()


# In[21]:


d3_clean[['votes', 'approx_cost(for two people)', 'rate']].kurt()


# In[22]:


mean_val = d3['rate'].mean()
median_val = d3['rate'].median()
sns.histplot(d3['rate'], bins=30, kde=True,edgecolor='white', linewidth=0.5)
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of restaurants")
plt.axvline(mean_val, color='red', linestyle='--', label=f"Mean: {mean_val:.2f}")
plt.axvline(median_val, color='green', linestyle='-', label=f"Median: {median_val:.2f}")
plt.legend()
plt.show()


# In[23]:


top_cuisines = d3['cuisines'].value_counts().head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=top_cuisines.values, y=top_cuisines.index)

plt.xlabel("Number of Restaurants")
plt.ylabel("Cuisines")
plt.title("Top 10 Most Popular Cuisines")

plt.show()


# In[24]:


top_locations = d3['location'].value_counts().head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=top_locations.values, y=top_locations.index)

plt.xlabel("Number of Restaurants")
plt.ylabel("Location")
plt.title("Top 10 Locations with Most Restaurants")

plt.show()


# In[25]:


plt.figure(figsize=(8,5))

sns.boxplot(x='online_order', y='rate', data=d3)

plt.xlabel("Online Order Available")
plt.ylabel("Rating")
plt.title("Impact of Online Ordering on Ratings")

plt.show()


# In[26]:


plt.figure(figsize=(8,5))

sns.scatterplot(
    x='approx_cost(for two people)',
    y='rate',
    data=d3

)

plt.title("Cost vs Rating")
plt.xlabel("Approx Cost for Two")
plt.ylabel("Rating")

plt.show()


# In[27]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))

corr = d3_clean[['votes', 'approx_cost(for two people)', 'rate']].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")

plt.title("Correlation Heatmap")
plt.show()


# In[28]:


from sklearn.preprocessing import LabelEncoder

d3_model = d3_clean.copy()

le = LabelEncoder()

d3_model['location_enc'] = le.fit_transform(d3_model['location'])
d3_model['cuisines_enc'] = le.fit_transform(d3_model['cuisines'])
d3_model['rest_type_enc'] = le.fit_transform(d3_model['rest_type'])

d3_model['online_order'] = d3_model['online_order'].map({'Yes': 1, 'No': 0})
d3_model['book_table'] = d3_model['book_table'].map({'Yes': 1, 'No': 0})


# In[29]:


features = [
    'votes',
    'approx_cost(for two people)',
    'rate',
    'location_enc',
    'cuisines_enc',
    'rest_type_enc',
    'online_order',
    'book_table'
]

X = d3_model[features]


# In[30]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[31]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)   
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.show()


# In[32]:


kmeans = KMeans(n_clusters=5, random_state=42)
d3_model['cluster'] = kmeans.fit_predict(X_scaled)


# In[33]:


plt.figure(figsize=(6,5))
plt.scatter(d3_model['rate'], d3_model['votes'], c=d3_model['cluster'], cmap='viridis')
plt.xlabel("Rating")
plt.ylabel("Votes")
plt.title("Restaurant Clusters")
plt.show()


# In[34]:


sample_data = d3_model.sample(2000, random_state=42)

plt.figure(figsize=(6,5))
plt.scatter(
    sample_data['rate'], 
    sample_data['votes'], 
    c=sample_data['cluster'], 
    cmap='viridis',
    alpha=0.6
)
plt.xlabel("Rating")
plt.ylabel("Votes")
plt.title("Restaurant Clusters (Sampled)")
plt.show()


# In[35]:


plt.figure(figsize=(6,4))
sns.countplot(x='cluster', data=d3_model)
plt.title("Number of Restaurants in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()


# In[36]:


cluster_summary = d3_model.groupby('cluster')[[
    'rate',
    'votes',
    'approx_cost(for two people)'
]].mean()

print(cluster_summary)


# In[37]:


d3_model.groupby('cluster')['cuisines'].agg(lambda x: x.mode()[0])


# In[42]:


def recommend_restaurants(data, cuisine, location, max_cost):

    filtered = data[
        (data['cuisines'].str.contains(cuisine, case=False, na=False)) &
        (data['location'].str.contains(location, case=False, na=False)) &
        (data['approx_cost(for two people)'] <= max_cost)
    ]

    if filtered.empty:
        return "No matching restaurants found"

    cluster_id = filtered.iloc[0]['cluster']

    recommendations = data[data['cluster'] == cluster_id]

    recommendations = recommendations[
        (recommendations['cuisines'].str.contains(cuisine, case=False, na=False)) &
        (recommendations['location'].str.contains(location, case=False, na=False)) &
        (recommendations['approx_cost(for two people)'] <= max_cost)
    ]

    recommendations = recommendations.sort_values(
        by=['rate', 'votes'], ascending=False
    )

    recommendations = recommendations.drop_duplicates(subset='name')

    recommendations = recommendations.head()

    return recommendations[['name', 'location', 'cuisines', 'rate', 'votes', 'approx_cost(for two people)']]


# In[44]:


cuisine = input("Enter cuisine: ")
location = input("Enter location: ")
max_cost = int(input("Enter max budget: "))

result = recommend_restaurants(d3_model, cuisine, location, max_cost)

print(result)


# In[ ]:




