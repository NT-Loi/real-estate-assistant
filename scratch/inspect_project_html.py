from bs4 import BeautifulSoup

with open("scratch/project_detail.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

print(f"Title: {soup.title.string if soup.title else 'No Title'}")

# List all divs that have classes related to box, spec, attr, or config
print("\n--- BOX / SPEC / ATTR DIVS ---")
classes = ["re__project-box-item", "re__prj-config-item", "re__project-info-item", "re__prj-overview-item", "re__prj-attribute-item"]
for c in classes:
    els = soup.find_all(class_=c)
    print(f"Class '{c}': found {len(els)} elements.")

# Search for the word "Chủ đầu tư"
print("\n--- TEXT SEARCH ---")
keywords = ["chủ đầu tư", "quy mô", "diện tích", "số tòa", "số căn hộ", "pháp lý", "mật độ xây dựng", "trạng thái", "mô tả", "tiện ích"]
for kw in keywords:
    el = soup.find(text=lambda t: t and kw.lower() in t.lower())
    if el:
        print(f"Keyword '{kw}' found! Parent tag: {el.parent.name}, class: {el.parent.get('class')}")
        # print some parents
        p = el.parent
        for idx in range(3):
            if p:
                print(f"  Parent {idx}: {p.name}, class: {p.get('class')}, text: {p.get_text(strip=True)[:150]}")
                p = p.parent
    else:
        print(f"Keyword '{kw}' NOT found!")
