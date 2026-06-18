from bs4 import BeautifulSoup

with open("scratch/listing_detail.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

words = ["Pháp lý", "Nội thất", "phòng ngủ", "toilet", "tầng", "Mặt tiền", "Đường vào"]
for w in words:
    print(f"\n--- SEARCHING FOR '{w}' ---")
    el = soup.find(text=lambda t: t and w.lower() in t.lower())
    if el:
        print(f"Found element: '{el}'")
        parent = el.parent
        for i in range(4):
            if parent:
                print(f"  Parent {i}: tag: {parent.name}, class: {parent.get('class')}")
                # print full html of parent 0 to see structure
                if i == 1:
                    print(f"  Parent 1 HTML: {parent}")
                parent = parent.parent
    else:
        print(f"'{w}' NOT found!")
