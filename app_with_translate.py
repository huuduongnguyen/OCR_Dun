import gradio as gr
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import fitz
import cv2
import numpy as np
from deep_translator import GoogleTranslator


def image_ocr(image_path, lang, font_name):
    font_paths = {
        "Vietnamese": r"C:\Users\ADMIN\Downloads\Inter,Open_Sans\Inter\Inter-VariableFont_opsz,wght.ttf",
        "Latin": r"D:\Github\PaddleOCR\doc\fonts\latin.ttf",
        "Japan": r"D:\Github\PaddleOCR\doc\fonts\japan.ttc",
        "Chinese": r"D:\Github\PaddleOCR\doc\fonts\chinese_cht.ttf",
        "Korean" : r"D:\Github\PaddleOCR\doc\fonts\korean.ttf"
    }
    font_path = font_paths[font_name]
    ocr = PaddleOCR(use_angle_cls=False, lang=lang, use_gpu=True)
    result = ocr.ocr(image_path, cls=True)
    annotated_image = Image.open(image_path).convert('RGB')
    boxes = [line[0] for line in result[0]]
    txts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]
    im_show = draw_ocr(annotated_image, boxes, txts, scores, font_path=font_path)
    im_show = Image.fromarray(im_show)
    text_output = "\n".join(txts)
    return im_show, text_output


def pdf_ocr(pdf_path, lang, font_name, page_num):
    font_paths = {
        "Vietnamese": r"C:\Users\ADMIN\Downloads\Inter,Open_Sans\Inter\Inter-VariableFont_opsz,wght.ttf",
        "Latin": r"D:\Github\PaddleOCR\doc\fonts\latin.ttf",
        "Japan": r"D:\Github\PaddleOCR\doc\fonts\japan.ttc",
        "Chinese": r"D:\Github\PaddleOCR\doc\fonts\chinese_cht.ttf",
        "Korean": r"D:\Github\PaddleOCR\doc\fonts\korean.ttf"
    }
    font_path = font_paths[font_name]
    ocr = PaddleOCR(use_angle_cls=True, lang=lang, page_num=page_num, use_gpu=True)
    result = ocr.ocr(pdf_path, cls=True)

    imgs = []
    with fitz.open(pdf_path) as pdf_file:
        for pg in range(0, page_num):
            page = pdf_file[pg]
            mat = fitz.Matrix(2, 2)
            pm = page.get_pixmap(matrix=mat, alpha=False)
            if pm.width > 2000 or pm.height > 2000:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            imgs.append(img)

    annotated_images = []
    text_output = ""
    for idx in range(len(result)):
        res = result[idx]
        if res is None:
            continue
        image = imgs[idx]
        boxes = [line[0] for line in res]
        txts = [line[1][0] for line in res]
        scores = [line[1][1] for line in res]
        im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
        annotated_images.append(Image.fromarray(im_show))
        text_output += "\n".join(txts) + "\n"

    return annotated_images, text_output


def translate_text(text, target_lang):
    translator = GoogleTranslator(target=target_lang)
    translated = translator.translate(text)
    return translated


with gr.Blocks() as demo:
    gr.Markdown("# Optical Character Recognition Web-based App")
    lang = gr.Radio(["vi", "en", "ch", "japan", "korean"], label="Select Language")
    font = gr.Dropdown(["Vietnamese","Latin", "Japan", "Chinese", "Korean"], label="Select Font")
    target_lang = gr.Dropdown(["en", "vi", "zh-CN", "ja", "ko"], label="Select Target Language")

    with gr.Tab("Image OCR"):
        image_input = gr.Image(type="filepath", label="Upload Image")
        image_output = gr.Image(label="OCR Result")
        image_text_output = gr.Textbox(label="OCR Text")
        translated_text_output = gr.Textbox(label="Translated Text")
        image_button = gr.Button("Submit")
        image_button.click(image_ocr, inputs=[image_input, lang, font], outputs=[image_output, image_text_output])
        translate_button = gr.Button("Translate")
        translate_button.click(translate_text, inputs=[image_text_output, target_lang], outputs=[translated_text_output])

    with gr.Tab("PDF OCR"):
        pdf_input = gr.File(type="filepath", label="Upload PDF")
        pdf_output = gr.Gallery(label="OCR Results")
        pdf_text_output = gr.Textbox(label="OCR Text")
        translated_pdf_text_output = gr.Textbox(label="Translated Text")
        page_num = gr.Slider(minimum=1, maximum=100, step=1, label="Select Page Number")
        pdf_button = gr.Button("Submit")
        pdf_button.click(pdf_ocr, inputs=[pdf_input, lang, font, page_num], outputs=[pdf_output, pdf_text_output])
        translate_pdf_button = gr.Button("Translate")
        translate_pdf_button.click(translate_text, inputs=[pdf_text_output, target_lang], outputs=[translated_pdf_text_output])

demo.launch(share=True)
