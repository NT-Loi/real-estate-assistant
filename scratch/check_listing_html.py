from bs4 import BeautifulSoup

with open("scratch/listing_detail.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")
print(f"Listing Title: {soup.title.string if soup.title else 'No Title'}")
print(f"HTML size: {len(html)} characters")

h1_el = soup.find("h1")
if h1_el:
    print(f"H1: {h1_el.get_text(strip=True)}")
else:
    print("H1 NOT found!")
