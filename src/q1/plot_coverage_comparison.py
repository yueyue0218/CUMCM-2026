"""Draw the analytical D=36 comparison, not simulation results.

Run: python src/q1/plot_coverage_comparison.py
Pillow is also installed with the repository's matplotlib dependency.
"""
from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[2]
scale = 2
im = Image.new('RGB', (1400*scale, 620*scale), 'white')
draw = ImageDraw.Draw(im)
font_path = 'C:/Windows/Fonts/arial.ttf'
if not Path(font_path).exists():
    font_path = 'DejaVuSans.ttf'
def font(size):
    return ImageFont.truetype(font_path, size*scale)
def text(x, y, message, size=24, fill='#203044'):
    draw.text((x*scale, y*scale), message, font=font(size), fill=fill, anchor='mm')
def point(center, p):
    return ((center[0]+8*p[0])*scale, (center[1]-8*p[1])*scale)
def circle(center, radius, color, dashed=False):
    x, y = center
    box = ((x-8*radius)*scale, (y-8*radius)*scale,
           (x+8*radius)*scale, (y+8*radius)*scale)
    if dashed:
        for angle in range(0, 360, 12):
            draw.arc(box, angle, angle+7, fill=color, width=2*scale)
    else:
        draw.ellipse(box, outline=color, width=3*scale)

text(700, 34, 'Same diameter, different minimum covering circles', 30)
shapes = [
    ((350, 285), [(x*18/math.sqrt(2), y*18/math.sqrt(2))
                  for x, y in [(-1,-1), (1,-1), (1,1), (-1,1)]],
     18, 'Square', 'r* = 18 m: a 20 m disk covers the region'),
    ((1050, 285), [(12*math.sqrt(3)*math.cos(math.radians(a)),
                    12*math.sqrt(3)*math.sin(math.radians(a)))
                   for a in [90, 210, 330]],
     12*math.sqrt(3), 'Equilateral triangle',
     'r* = 20.784610 m: no 20 m disk can cover it'),
]
for center, vertices, radius, title, result in shapes:
    pixels = [point(center, p) for p in vertices]
    draw.polygon(pixels, fill='#E6F2F7', outline='#617990', width=2*scale)
    circle(center, 20, '#929AA3', dashed=True)
    circle(center, radius, '#16777E')
    for x, y in pixels:
        draw.ellipse((x-4*scale,y-4*scale,x+4*scale,y+4*scale),fill='#203044')
    x, y = point(center, (0,0))
    draw.ellipse((x-3*scale,y-3*scale,x+3*scale,y+3*scale),fill='#16777E')
    text(center[0], 83, title, 26)
    text(center[0], 477, 'D = 36 m', 24)
    text(center[0], 515, result, 20)
text(700, 573, 'Solid teal: minimum covering circle    Dashed gray: radius 20 m', 22)
out = root/'results/figures/q1_same_diameter_coverage.png'
out.parent.mkdir(parents=True, exist_ok=True)
im.save(out, dpi=(300,300))
print(out)
