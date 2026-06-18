from bs4 import BeautifulSoup

with open("scratch/listing_detail.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

print("--- INSPECTING re__pr-short-info-item ---")
items = soup.find_all(class_="re__pr-short-info-item")
print(f"Found {len(items)} items:")
for idx, item in enumerate(items):
    print(f"\nItem {idx+1}: innerHTML = {item}")
    for child in item.find_all(class_=True):
        print(f"  Child class: {child.get('class')}, text: '{child.get_text(strip=True)}'")
