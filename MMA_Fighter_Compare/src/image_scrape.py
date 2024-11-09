import requests
from bs4 import BeautifulSoup
import os
import re
import pandas as pd

def read_in_fighter_names():
    # Load the CSV file into a DataFrame
    df = pd.read_csv('MMA_Fighter_Compare\\data\\fighter_stats.csv')

    # Extract the 'name' column as a list
    names = df['name'].tolist()

    # Display the names
    return names

def format_fighter_name(name):
    # Split the name into first and last names
    first_name, last_name = name.split()
    
    # Format the string as LASTNAME_FIRSTNAME_L in uppercase
    formatted_name = f"{last_name.upper()}_{first_name.upper()}_L"
    
    return formatted_name

def sanitize_filename(filename):
    # Remove invalid characters for filenames on most OSes
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def download_image(img_url, save_folder="images", img_name=None):

    os.makedirs(save_folder, exist_ok=True)  # Create folder if it doesn't exist

    if img_name is None:
        # Use the last part of the URL as the filename, sanitized
        img_name = sanitize_filename(img_url.split("/")[-1].split("?")[0])

    img_path = os.path.join(save_folder, img_name)
    response = requests.get(img_url)

    if response.status_code == 200:
        with open(img_path, "wb") as file:
            file.write(response.content)
        print(f"Downloaded: {img_path}")
    else:
        print(f"Failed to download: {img_url}")

# Main scraping function
def scrape_images_from_html(html_content, fighter_name, save_folder="images"):

    soup = BeautifulSoup(html_content, "html.parser")
    img_tags = soup.find_all("img")
    fighter_name_html = format_fighter_name(fighter_name)
    fighter_img = None

    for img in img_tags:

        # Check if the search string is in the 'src' or 'alt' attribute
        if fighter_name_html in img.get('src', '') or fighter_name_html in img.get('alt', ''):
            fighter_img = img
            break  # Stop after finding the first match
    
    if fighter_img is not None:
        img_url = fighter_img.get("src")
        download_image(img_url, save_folder)

# Sample usage
if __name__ == "__main__":

    fighter_names = read_in_fighter_names()

    for name in fighter_names:

        name_f = name.lower().replace(" ", "-")
        url = f"https://www.ufc.com/athlete/{name_f}"  # Replace with the page URL you want to scrape
        response = requests.get(url)

        if response.status_code == 200:
            scrape_images_from_html(response.text, name)
        else:
            print("Failed to retrieve webpage")
