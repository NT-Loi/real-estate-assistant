from bs4 import BeautifulSoup

with open("scratch/project_detail.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

print("--- INSPECTING TABLES AND SPEC ROWS ---")
# Let's find all tr elements
rows = soup.find_all("tr")
print(f"Found {len(rows)} tr elements in the page:")
for idx, r in enumerate(rows):
    cells = r.find_all(["td", "th"])
    cell_info = [f"{c.name}[class={c.get('class')}]: '{c.get_text(strip=True)}'" for c in cells]
    print(f"Row {idx+1}: " + " | ".join(cell_info))

# Let's inspect re__project-box-item inner structure
print("\n--- INSPECTING re__project-box-item ---")
box_items = soup.find_all(class_="re__project-box-item")
for idx, item in enumerate(box_items):
    print(f"Box item {idx+1}: innerHTML = {item}")
