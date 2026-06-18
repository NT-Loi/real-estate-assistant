from bs4 import BeautifulSoup

with open("scratch/listing_detail.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

print("--- INSPECTING SHORT INFO ITEMS ---")
short_items = soup.select(".re__pr-short-info-item")
print(f"Found {len(short_items)} short info items:")
for item in short_items:
    print(item.prettify())

print("\n--- INSPECTING ALL div.re__row-item OR OTHER INFO BLOCKS ---")
for tag in soup.find_all(class_=lambda x: x and ("info" in x or "attr" in x or "specs" in x)):
    # Let's filter out very large containers
    text = tag.get_text(strip=True)
    if len(text) < 300 and tag.name not in ["script", "style"]:
        print(f"Tag: {tag.name}, Class: {tag.get('class')}, Text: '{text}'")
