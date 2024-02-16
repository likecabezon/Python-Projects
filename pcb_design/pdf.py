from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os
import subprocess
from reportlab.lib.colors import black
from reportlab.pdfgen import canvas

def create_svg_file(width_mm, height_mm, output_file):
    # Convert millimeters to SVG units (1mm = 3.543307 SVG units)
    width_svg_units = width_mm * 3.7792894935
    height_svg_units = height_mm * 3.7792894935

    # Generate the SVG content
    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width_svg_units:.2f}" height="{height_svg_units:.2f}">
    <rect width="{width_svg_units:.2f}" height="{height_svg_units:.2f}" fill="black" />
</svg>
"""

    # Write the content to the output file
    with open(output_file, "w") as f:
        f.write(svg_content)


if __name__ == "__main__":
    width_mm = 10
    height_mm = 10
    output_file = "black_rectangle.svg"

    create_svg_file(width_mm, height_mm, output_file)
    print(f"SVG file '{output_file}' generated successfully.")