

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Function to download an image from a URL and save it to a local file
# Function to download an image from a URL and save it to a local file
def download_image(url, directory):
    # Check if the URL is a data URI
    if url.startswith('data:'):
        print(f"Skipped data URI image: {url}")
        return

    response = requests.get(url)
    if response.status_code == 200:
        file_name = os.path.join(directory, os.path.basename(url))
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {file_name}")
    else:
        print(f"Failed to download: {url}")



# URL of the webpage you want to crawl
url = 'https://gocookyummy.com/'  # Replace with the URL of your target website

# Send a GET request to the URL
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')

    # Directory where you want to save the downloaded images
    save_directory = 'downloaded_images'
    create_directory(save_directory)

    # Find all image tags in the HTML
    img_tags = soup.find_all('img')

    for img_tag in img_tags:
        # Get the image URL and make it absolute if it's a relative URL
        img_url = img_tag.get('src')
        if img_url and not img_url.startswith('http'):
            img_url = urljoin(url, img_url)

        # Download the image
        if img_url:
            download_image(img_url, save_directory)
else:
    print(f"Failed to fetch the webpage: {url}")
