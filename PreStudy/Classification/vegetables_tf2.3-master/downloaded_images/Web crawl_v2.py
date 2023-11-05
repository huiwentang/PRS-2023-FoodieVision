import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to download an image from a URL and save it to a local file
def download_image(url, directory):
    response = requests.get(url)
    if response.status_code == 200:
        file_name = os.path.join(directory, os.path.basename(url))
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {file_name}")
    else:
        print(f"Failed to download: {url}")

# Function to crawl a webpage and download all images
def crawl_and_download_images(url, save_directory):
    # Send a GET request to the URL
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all image tags in the HTML
        img_tags = soup.find_all('img')

        for img_tag in img_tags:
            # Get the image URL and make it absolute if it's a relative URL
            img_url = img_tag.get('src')
            if img_url:
                img_url = urljoin(url, img_url)

                # Check if the URL is a valid image file (you can add more extensions if needed)
                if re.search(r'\.(jpg|jpeg|png|gif)$', img_url, re.IGNORECASE):
                    download_image(img_url, save_directory)
    else:
        print(f"Failed to fetch the webpage: {url}")

# URL of the webpage you want to crawl
# start_url = 'https://gocookyummy.com/'  # Replace with the URL of your target website

start_url = 'https://www.istockphoto.com/cs/search/2/image?mediatype=photography&phrase=food&sort=mostpopular&irgwc=1&cid=IS&utm_medium=affiliate&utm_source=Jakub%20Kapusnak&clickid=WHrRjo3FoxyPTApVQZxALQK5UkFWNRSUZSlO280&utm_term=POPUP&utm_campaign=&utm_content=258824&irpid=1263831'
# Directory where you want to save the downloaded images
save_directory = 'downloaded_images'
create_directory(save_directory)

# Crawl and download images
crawl_and_download_images(start_url, save_directory)
