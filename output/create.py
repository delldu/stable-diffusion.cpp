import os
import string

# 设置当前目录
directory = '.'  # 点代表当前目录

# 定义HTML模板
html_template = string.Template("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Gallery</title>
    <style>
        .image-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            padding: 10px;
        }
        .image-gallery img {
            width: 100%;
            height: auto;
            object-fit: cover;
        }
    </style>
</head>
<body>
    <div class="image-gallery">
        $image_tags
    </div>
</body>
</html>
""")

# 遍历目录，找到所有图片文件
image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
image_files = sorted(image_files)

# 生成图片标签
image_tags = '\n'.join([f'<a href="{f}"><img src="{f}" alt="Image {i}" style="width:200px;height:200px;"></a>' for i, f in enumerate(image_files)])

# 使用Template填充HTML模板
filled_html = html_template.substitute(image_tags=image_tags)

# 将生成的HTML写入文件
with open('index.html', 'w', encoding='utf-8') as file:
    file.write(filled_html)

# 可选：使用webbrowser模块打开生成的HTML文件
# import webbrowser
# webbrowser.open('file://' + os.path.realpath('index.html'))
