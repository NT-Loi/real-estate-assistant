from bs4 import BeautifulSoup

with open("scratch/project_detail.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")
print(f"Title: {soup.title.string if soup.title else 'No Title'}")
print(f"HTML size: {len(html)} characters")

# Find any headings (h1, h2, h3)
print("\n--- HEADINGS ---")
for h in ["h1", "h2", "h3"]:
    els = soup.find_all(h)
    print(f"Found {len(els)} {h} elements:")
    for el in els[:5]:
        print(f"  {h}: class={el.get('class')}, text={el.get_text(strip=True)[:100]}")

# Find any elements with classes containing 'project' or 'prj'
print("\n--- CLASSES WITH 'project' or 'prj' ---")
seen_classes = set()
for el in soup.find_all(class_=True):
    for c in el.get("class"):
        if "project" in c.lower() or "prj" in c.lower():
            if c not in seen_classes:
                seen_classes.add(c)
                print(f"Found class: {c}")

# Let's search for some text that we would expect, like "Chủ đầu tư" or "Quy mô" or "Pháp lý"
print("\n--- TEXT SEARCH ---")
keywords = ["chủ đầu tư", "quy mô", "diện tích", "pháp lý", "vinhomes"]
for kw in keywords:
    el = soup.find(text=lambda t: t and kw.lower() in t.lower())
    if el:
        print(f"Keyword '{kw}' found! Parent tag: {el.parent.name}, class: {el.parent.get('class')}")
        # print some surrounding HTML
        print(f"  Surrounding text: {el.parent.get_text(strip=True)[:200]}")
    else:
        print(f"Keyword '{kw}' NOT found!")
