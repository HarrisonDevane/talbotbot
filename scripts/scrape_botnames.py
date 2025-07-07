import requests
from bs4 import BeautifulSoup

def get_bots_page_1():
    url = "https://lichess.org/player/bots?page=1"
    print(f"Fetching {url} ...")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch page 1: status code {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    bot_links = soup.select('a[href^="/@"]')
    
    bots = []
    for link in bot_links:
        bot_name = link.text.strip()
        if bot_name and bot_name not in bots:
            bots.append(bot_name)
    
    print(f"Found {len(bots)} bots on page 1.")
    return bots

if __name__ == "__main__":
    bots = get_bots_page_1()
    print(bots)
