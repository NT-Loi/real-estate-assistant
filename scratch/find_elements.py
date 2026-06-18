from bs4 import BeautifulSoup

with open("scratch/listing_detail.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

# Search for "Đặc điểm bất động sản" heading
print("--- SEARCHING 'Đặc điểm bất động sản' ---")
heading = soup.find(text=lambda t: t and "Đặc điểm bất động sản" in t)
if heading:
    print(f"Found heading: '{heading}'")
    parent = heading.parent
    for _ in range(4):
        if parent:
            print(f"Parent tag: {parent.name}, classes: {parent.get('class')}")
            parent = parent.parent
            
    # Print next sibling elements
    print("\nNext siblings of parent:")
    heading_parent = heading.parent
    for sibling in heading_parent.next_siblings:
        if sibling.name:
            print(f"Sibling: {sibling.name}, class: {sibling.get('class')}, text preview: {sibling.get_text(strip=True)[:150]}")
            # Print the inner structure
            for child in sibling.find_all(class_=True):
                print(f"  Child tag: {child.name}, class: {child.get('class')}, text: {child.get_text(strip=True)}")
else:
    print("Heading 'Đặc điểm bất động sản' NOT found!")

print("\n--- SEARCHING 'Thông tin dự án' ---")
project_heading = soup.find(text=lambda t: t and "Thông tin dự án" in t)
if project_heading:
    print(f"Found project heading: '{project_heading}'")
    parent = project_heading.parent
    for _ in range(4):
        if parent:
            print(f"Parent tag: {parent.name}, classes: {parent.get('class')}")
            parent = parent.parent
            
    print("\nNext siblings of project parent:")
    proj_parent = project_heading.parent
    for sibling in proj_parent.next_siblings:
        if sibling.name:
            print(f"Sibling: {sibling.name}, class: {sibling.get('class')}, text preview: {sibling.get_text(strip=True)[:150]}")
            for child in sibling.find_all(class_=True):
                print(f"  Child tag: {child.name}, class: {child.get('class')}, text: {child.get_text(strip=True)}")
else:
    print("Heading 'Thông tin dự án' NOT found!")
