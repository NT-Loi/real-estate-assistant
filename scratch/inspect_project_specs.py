from bs4 import BeautifulSoup

with open("scratch/project_detail.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

print("--- INSPECTING ALL CLASSES WITH project-info-details ---")
info_details = soup.find_all(class_=lambda x: x and "project-info-details" in x)
print(f"Found {len(info_details)} elements:")
for el in info_details:
    print(f"  Class: {el.get('class')}, Text: '{el.get_text(strip=True)}'")
    # Print children if any
    for c in el.children:
        if c.name:
            print(f"    Child: {c.name}, Class: {c.get('class')}, Text: '{c.get_text(strip=True)}'")

print("\n--- INSPECTING re__project-toogle-box OR OTHER COLLAPSIBLE BOXES ---")
toogle_boxes = soup.find_all(class_=lambda x: x and ("toogle" in x or "collapse" in x or "attr" in x))
for box in toogle_boxes:
    text = box.get_text(strip=True)
    if len(text) < 500:
        print(f"Class: {box.get('class')}, Text: '{text}'")
