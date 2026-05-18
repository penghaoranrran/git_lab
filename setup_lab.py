import os
from PIL import Image, ImageDraw, ImageFont

def create_images():
    if not os.path.exists('images'):
        os.makedirs('images')
    
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
    
    for i in range(1, 7):
        # Create a new image with a solid color
        img = Image.new('RGB', (200, 200), color=colors[i-1])
        
        # Initialize ImageDraw
        d = ImageDraw.Draw(img)
        
        # Add text to the image
        text = f"Image {i}"
        # Use default font
        try:
            # Try to load a font, fall back to default if necessary
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        # Get text position (centering is rough with default font but okay)
        d.text((50, 90), text, fill=(0, 0, 0), font=font)
        
        img.save(f'images/image{i}.jpg')
        print(f"Created images/image{i}.jpg")

def create_html():
    html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>计算机网络 Lab3</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; }
        .info { margin: 20px; padding: 20px; border: 1px solid #ccc; display: inline-block; }
        .gallery { display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin-top: 20px; }
        .gallery img { border: 2px solid #333; }
    </style>
</head>
<body>
    <h1>实验3：Web页面</h1>
    
    <div class="info">
        <h2>基本信息</h2>
        <p><strong>专业：</strong>信息安全</p>
        <p><strong>学号：</strong>2313314</p>
        <p><strong>姓名：</strong>彭浩然</p>
    </div>

    <h2>图片展示</h2>
    <div class="gallery">
        <img src="images/image1.jpg" alt="Image 1">
        <img src="images/image2.jpg" alt="Image 2">
        <img src="images/image3.jpg" alt="Image 3">
        <img src="images/image4.jpg" alt="Image 4">
        <img src="images/image5.jpg" alt="Image 5">
        <img src="images/image6.jpg" alt="Image 6">
    </div>
</body>
</html>
"""
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("Created index.html")

if __name__ == "__main__":
    create_images()
    create_html()
