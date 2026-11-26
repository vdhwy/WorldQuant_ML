import requests
from bs4 import BeautifulSoup

url = "https://www.quantinsti.com/blog/top-python-libraries-for-quant-finance"

response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

for h3 in soup.find_all("h3"):
    print(h3.text.strip())