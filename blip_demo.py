import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO

class BLIPDemo:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
    
    def load_image(self, image_path_or_url):
        if image_path_or_url.startswith("http"):
            img = Image.open(BytesIO(requests.get(image_path_or_url).content)).convert("RGB")
        else:
            img = Image.open(image_path_or_url).convert("RGB")
        return img

    def generate_caption(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)
    
    def vqa(self, image, question):
        # BLIP base is not fine-tuned for VQA, but this shows the idea.
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":

    sample_img_url = "https://farm3.staticflickr.com/2746/4498258642_1f93655c81_z.jpg"

    blip = BLIPDemo()

    print("Downloading sample image...")
    image = blip.load_image(sample_img_url)

    print("\n== Captioning ==")
    caption = blip.generate_caption(image)
    print("Caption:", caption)

    print("\n== VQA Example ==")
    question = "What is the man doing?"
    answer = blip.vqa(image, question)
    print(f"Q: {question}\nA: {answer}")

    # Save image with caption for screenshot
    from PIL import ImageDraw, ImageFont
    img_with_caption = image.copy()
    draw = ImageDraw.Draw(img_with_caption)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), caption, font=font, fill="white")
    img_with_caption.save("captioned_sample.jpg")
    print("\nSaved: captioned_sample.jpg (for screenshot)")
