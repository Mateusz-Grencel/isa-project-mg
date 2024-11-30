import requests
from bs4 import BeautifulSoup
import pandas as pd
import nomic
from nomic import atlas, AtlasDataset
from top_serp import search_top_30
from keys import Nomic_API_Key

# Function to fetch page title and meta description
def fetch_page_metadata(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.string if soup.title else "Title not found"

        meta_desc = soup.find("meta", attrs={"name": "description"})
        meta_description = meta_desc.get('content') if meta_desc else "Meta description not found"

        return title, meta_description
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page metadata: {e}")
        return None, None

# Function to validate URL
def validate_url(url):
    if not url.startswith(('http://', 'https://')):
        raise ValueError("Invalid URL: It must start with 'http://' or 'https://'.")

# Input URL and search phrase
url = input("Enter your website URL: ")
phrase = input("Enter the keyword related to your website: ")

# Validate the URL
try:
    validate_url(url)
except ValueError as e:
    print(e)
    exit(1)  # Exit if the URL is invalid

# Fetch page metadata
title, meta_description = fetch_page_metadata(url)

# Fetch search results for the phrase
try:
    results = search_top_30(phrase)
except Exception as e:
    print(f"Error fetching search results: {e}")
    results = []

# Initialize Nomic API
nomic.login(Nomic_API_Key)

# Create DataFrame from search results
df = pd.DataFrame(results)

# Add our URL, title, and meta description to the DataFrame
if title and meta_description:
    df.loc[len(df)] = [len(df) + 1, title, url, meta_description]

# Save DataFrame to CSV
df.to_csv('top_results.csv', index=False)

# Print DataFrame columns
print("Columns in DataFrame: ", df.columns)

# Rename columns for easier reference
df["position"] = df["pozycja"]
df["title"] = df["tytuł"]
df["link"] = df["link"]
df["meta_description"] = df["metaopis"]

# Required columns
required_columns = {"position", "title", "link", "meta_description"}

# Check if all required columns are present
if required_columns.issubset(df.columns):
    filtered_df = df[['position', 'title', 'link', 'meta_description']]

    if not filtered_df.empty:
        # Combine columns into 'indexed_field'
        filtered_df['indexed_field'] = filtered_df.apply(
            lambda row: f"{row['position']} {row['title']} {row['link']} {row['meta_description']}", axis=1
        )

        filtered_df = filtered_df.reset_index(drop=True)

        # Pass the data to Atlas
        try:
            dataset = atlas.map_data(data=filtered_df, indexed_field='indexed_field')
            print(dataset)
        except Exception as e:
            print(f"Error mapping data to Atlas: {e}")
    else:
        print("No data to process.")
else:
    missing_columns = required_columns - set(df.columns)
    print(f"Missing columns: {missing_columns}")

# Load dataset
try:
    print("Please wait for an email confirmation regarding the creation of the dataset (up to 10 minutes). "
          "You can check your email for the confirmation. After receiving it, please proceed to the 'sec.py' file.")
except Exception as e:
    print(f"Error loading dataset or map: {e}")
