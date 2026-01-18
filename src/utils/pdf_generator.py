import markdown
from xhtml2pdf import pisa
import os
import re
from pathlib import Path
from typing import Optional

def resolve_image_path(img_path: str, base_dir_abs: Optional[str]) -> Optional[str]:
    if not img_path:
        return None

    if os.path.isabs(img_path) and os.path.exists(img_path):
        return img_path

    candidates = []
    if base_dir_abs:
        candidates.append(os.path.join(base_dir_abs, img_path))
        basename = os.path.basename(img_path)
        candidates.append(os.path.join(base_dir_abs, "artifacts", "plots", basename))
        candidates.append(os.path.join(base_dir_abs, "work", "artifacts", "plots", basename))
        candidates.append(os.path.join(base_dir_abs, "sandbox", "downloaded_artifacts", "plots", basename))
    else:
        candidates.append(os.path.abspath(img_path))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None

def convert_report_to_pdf(
    markdown_content: str,
    output_filename: str = "final_report.pdf",
    base_dir: Optional[str] = None,
) -> bool:
    """
    Converts markdown content to a PDF file using xhtml2pdf.
    Resolves image paths and adds basic styling.
    """
    try:
        base_dir_abs = os.path.abspath(base_dir) if base_dir else None
        # 1. Extract and Remove Images from Markdown
        # Find all images: ![Alt](path)
        images = re.findall(r'!\[(.*?)\]\((.*?)\)', markdown_content)
        
        # Remove images from markdown text so we can control their placement manually
        markdown_text_clean = re.sub(r'!\[(.*?)\]\((.*?)\)', '', markdown_content)
        
        # FALLBACK: If no images in markdown, scan known plot folders
        if not images:
            if base_dir_abs:
                candidate_dirs = [
                    os.path.join(base_dir_abs, "static", "plots"),
                    os.path.join(base_dir_abs, "artifacts", "plots"),
                    os.path.join(base_dir_abs, "work", "artifacts", "plots"),
                ]
            else:
                candidate_dirs = [
                    os.path.abspath(os.path.join("static", "plots")),
                    os.path.abspath(os.path.join("artifacts", "plots")),
                    os.path.abspath(os.path.join("work", "artifacts", "plots")),
                ]
            supported_ext = ('.png', '.jpg', '.jpeg')
            for plots_dir in candidate_dirs:
                if not os.path.exists(plots_dir):
                    continue
                plot_files = sorted([f for f in os.listdir(plots_dir) if f.lower().endswith(supported_ext)])
                for f in plot_files:
                    full_path = os.path.join(plots_dir, f)
                    images.append((f, full_path))
        
        # 2. Convert Cleaned Markdown to HTML
        html_content = markdown.markdown(markdown_text_clean, extensions=['tables', 'fenced_code'])
        
        # 3. Construct Image Grid (2-Column Table)
        image_grid_html = ""
        image_grid_html += "<h3>Visual Analysis</h3>"
        
        resolved_images = []
        for alt_text, img_path in images:
            abs_path = resolve_image_path(img_path, base_dir_abs)
            if abs_path:
                resolved_images.append((alt_text, abs_path))

        if resolved_images:
            image_grid_html += '<table style="width: 100%; border: none;">'
            
            for i, (alt_text, abs_path) in enumerate(resolved_images):
                # Start row for even indices (0, 2, 4...)
                if i % 2 == 0:
                    image_grid_html += '<tr>'

                # Create Cell
                img_src = Path(abs_path).resolve().as_posix()
                image_grid_html += f'''
                    <td style="width: 50%; padding: 5px; vertical-align: top; border: none;">
                        <div style="text-align: center;">
                            <img src="{img_src}" style="width: 500px; height: auto;" />
                            <p style="font-size: 8pt; color: #666;">{alt_text}</p>
                        </div>
                    </td>
                '''
                
                # End row for odd indices (1, 3, 5...) OR if it's the last image
                if i % 2 == 1 or i == len(resolved_images) - 1:
                    image_grid_html += '</tr>'
            
            image_grid_html += '</table>'
        else:
            image_grid_html += '<p style="color: #666; font-style: italic;">No plots were generated.</p>'

        # 4. Add CSS Styling and Assemble
        styled_html = f"""
        <html>
        <head>
            <style>
                @page {{
                    size: A4;
                    margin: 1.5cm;
                }}
                body {{
                    font-family: Helvetica, Arial, sans-serif;
                    font-size: 11pt;
                    line-height: 1.4;
                    color: #333;
                }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
                h2 {{ color: #e74c3c; margin-top: 20px; }}
                h3 {{ color: #34495e; margin-top: 15px; border-bottom: 1px solid #eee; }}
                code {{ background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-family: Courier New, monospace; }}
                pre {{ background-color: #f8f8f8; border: 1px solid #ddd; padding: 10px; overflow-x: auto; }}
                
                /* Data Tables */
                table.data {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                table.data th, table.data td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
                table.data th {{ background-color: #f2f2f2; font-weight: bold; }}

                .footer {{ position: fixed; bottom: 0; width: 100%; text-align: center; font-size: 9pt; color: #aaa; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>The Insight Foundry - Executive Report</h1>
            </div>
            
            {html_content}
            
            <div class="visualizations">
                {image_grid_html}
            </div>

            <div class="footer">
                Generated by Gemini Agents | Automated Business Intelligence
            </div>
        </body>
        </html>
        """
        
        output_path = output_filename
        if base_dir_abs and not os.path.isabs(output_filename):
            output_path = os.path.join(base_dir_abs, output_filename)
        # 4. Write to PDF
        with open(output_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(
                styled_html, dest=pdf_file
            )
            
        if pisa_status.err:
            print(f"PDF generation error: {pisa_status.err}")
            return False
            
        print(f"PDF generated successfully: {output_path}")
        return True

    except Exception as e:
        print(f"Failed to generate PDF: {e}")
        return False
