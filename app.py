import gradio as gr
import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
    StableDiffusion3Pipeline
)
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import hex2color
import stone
import os

access_token = os.getenv("HF_Token")


from huggingface_hub import login
login(token = access_token)


# Define model initialization functions
def load_model(model_name):
    """
    Load a StableDiffusionXLPipeline from a single file.

    Args:
        model_name (str): The name of the model.

    Returns:
        StableDiffusionXLPipeline: The loaded pipeline.

    Raises:
        ValueError: If the model name is unknown.

    """
    if model_name == "sinteticoXL":
        pipeline = StableDiffusionXLPipeline.from_single_file(
            "https://huggingface.co/lucianosb/sinteticoXL-models/blob/main/sinteticoXL_v1dot2.safetensors",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")
    elif model_name == "sinteticoXL_Prude":
        pipeline = StableDiffusionXLPipeline.from_single_file(
            "https://huggingface.co/lucianosb/sinteticoXL-models/blob/main/sinteticoXL_prude_v1dot2.safetensors",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")
    else:
        raise ValueError("Unknown model name")
    return pipeline

# Initialize the default model
default_model = "sinteticoXL"
pipeline_text2image = load_model(default_model)


def getimgen(prompt, model_name):
    """
    This function generates an image based on the prompt and the specified model name.

    Args:
        prompt (str): The input prompt for generating the image.
        model_name (str): The name of the model to use for image generation.

    Returns:
        Image: The generated image based on the prompt and model.
    """
    if model_name == "sinteticoXL":
        return pipeline_text2image(prompt=prompt, guidance_scale=6.0, num_inference_steps=20).images[0]
    elif model_name == "sinteticoXL_Prude":
        return pipeline_text2image(prompt=prompt, guidance_scale=6.0, num_inference_steps=20).images[0]

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")


def blip_caption_image(image, prefix):
    """
    This function generates a caption for the input image based on the provided prefix.
    
    Args:
        image: The input image for which the caption is generated.
        prefix: The prefix used for caption generation.
        
    Returns:
        str: The generated caption for the input image.
    """
    inputs = blip_processor(image, prefix, return_tensors="pt").to("cuda", torch.float16)
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def genderfromcaption(caption):
    """
    A function that determines the gender based on the input caption.

    Args:
        caption (str): The caption for which the gender needs to be determined.

    Returns:
        str: The gender identified from the caption (either "Man", "Woman", or "Unsure").
    """
    cc = caption.split()
    if "man" in cc or "boy" in cc:
        return "Man"
    elif "woman" in cc or "girl" in cc:
        return "Woman"
    return "Unsure"

def genderplot(genlist):    
    """
    A function that plots gender-related data based on the given list of genders.

    Args:
        genlist (list): A list of gender labels ("Man", "Woman", or "Unsure").

    Returns:
        fig: A matplotlib figure object representing the gender plot.
    """
    order = ["Man", "Woman", "Unsure"]
    words = sorted(genlist, key=lambda x: order.index(x))
    colors = {"Man": "lightgreen", "Woman": "darkgreen", "Unsure": "lightgrey"}
    word_colors = [colors[word] for word in words]
    fig, axes = plt.subplots(2, 5, figsize=(5,5))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=word_colors[i]))
    return fig

def age_detector(image):
    """
    A function that detects the age from an image.

    Args:
        image: The input image for age detection.

    Returns:
        str: The detected age label from the image.
    """
    pipe = pipeline('image-classification', model="dima806/faces_age_detection", device=0)
    result = pipe(image)
    max_score_item = max(result, key=lambda item: item['score'])
    return max_score_item['label']

def ageplot(agelist):
    """
    A function that plots age-related data based on the given list of age categories.

    Args:
        agelist (list): A list of age categories ("YOUNG", "MIDDLE", "OLD").

    Returns:
        fig: A matplotlib figure object representing the age plot.
    """
    order = ["YOUNG", "MIDDLE", "OLD"]
    words = sorted(agelist, key=lambda x: order.index(x))
    colors = {"YOUNG": "skyblue", "MIDDLE": "royalblue", "OLD": "darkblue"}
    word_colors = [colors[word] for word in words]
    fig, axes = plt.subplots(2, 5, figsize=(5,5))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=word_colors[i]))
    return fig

def is_nsfw(image):
    """
    A function that checks if the input image is not safe for work (NSFW) by classifying it using 
    an image classification pipeline and returning the label with the highest score.
    
    Args:
        image: The input image to be classified.
        
    Returns:
        str: The label of the NSFW category with the highest score.
    """
    classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
    result = classifier(image)
    max_score_item = max(result, key=lambda item: item['score'])
    return max_score_item['label']

def nsfwplot(nsfwlist):
    """
    Generates a plot of NSFW categories based on a list of NSFW labels.

    Args:
        nsfwlist (list): A list of NSFW labels ("normal" or "nsfw").

    Returns:
        fig: A matplotlib figure object representing the NSFW plot.

    Raises:
        None

    This function takes a list of NSFW labels and generates a plot with a grid of 2 rows and 5 columns. 
    Each label is sorted based on a predefined order and assigned a color. The plot is then created using matplotlib, 
    with each cell representing an NSFW label. The color of each cell is determined by the corresponding label's color. 
    The function returns the generated figure object.
    """
    order = ["normal", "nsfw"]
    words = sorted(nsfwlist, key=lambda x: order.index(x))
    colors = {"normal": "mistyrose", "nsfw": "red"}
    word_colors = [colors[word] for word in words]
    fig, axes = plt.subplots(2, 5, figsize=(5,5))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=word_colors[i]))
    return fig

def skintoneplot(hex_codes):
    """
    Generates a plot of skin tones based on a list of hexadecimal color codes.

    Args:
        hex_codes (list): A list of hexadecimal color codes.

    Returns:
        fig: A matplotlib figure object representing the skin tone plot.

    Raises:
        None

    This function takes a list of hexadecimal color codes and generates a plot with a grid of 2 rows and 5 columns. 
    Each color code is converted to its corresponding RGB value and then the luminance value is calculated using the 
    formula: luminance = 0.299 * R + 0.587 * G + 0.114 * B. The colors are then sorted based on their luminance values 
    in descending order and assigned to the corresponding cells in the plot. The plot is created using matplotlib, 
    with each cell representing a skin tone. The color of each cell is determined by the corresponding skin tone color. 
    The function returns the generated figure object.
    """
    hex_codes = [code for code in hex_codes if code is not None]
    rgb_values = [hex2color(hex_code) for hex_code in hex_codes]
    luminance_values = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in rgb_values]
    sorted_hex_codes = [code for _, code in sorted(zip(luminance_values, hex_codes), reverse=True)]
    fig, axes = plt.subplots(2, 5, figsize=(5,5))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if i < len(sorted_hex_codes):
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=sorted_hex_codes[i]))
    return fig

def generate_images_plots(prompt, model_name):
    """
    This function generates images, extracts information like genders, skintones, ages, and nsfw labels from the images, 
    and returns a tuple containing the images and plots for skintones, genders, ages, and nsfw labels. 

    Args:
        prompt (str): The prompt for generating images.
        model_name (str): The name of the model.

    Returns:
        tuple: A tuple containing the images generated, skintone plot, gender plot, age plot, and nsfw plot.

    Raises:
        None
    """
    global pipeline_text2image
    pipeline_text2image = load_model(model_name)
    foldername = "temp"
    Path(foldername).mkdir(parents=True, exist_ok=True)
    images = [getimgen(prompt, model_name) for _ in range(10)]
    genders = []
    skintones = []
    ages = []
    nsfws = []
    for image, i in zip(images, range(10)):
        prompt_prefix = "photo of a "
        caption = blip_caption_image(image, prefix=prompt_prefix)
        age = age_detector(image)
        nsfw = is_nsfw(image)
        image.save(f"{foldername}/image_{i}.png")
        try:
            skintoneres = stone.process(f"{foldername}/image_{i}.png", return_report_image=False)
            tone = skintoneres['faces'][0]['dominant_colors'][0]['color']
            skintones.append(tone)
        except:
            skintones.append(None)
        genders.append(genderfromcaption(caption))
        ages.append(age)
        nsfws.append(nsfw)
    return images, skintoneplot(skintones), genderplot(genders), ageplot(ages), nsfwplot(nsfws)

with gr.Blocks(title="Bias detection in SinteticoXL Models") as demo:
    gr.Markdown("# Bias detection in SinteticoXL Models")
    gr.Markdown('''
In this demo, we explore the potential biases in text-to-image models by generating multiple images based on user prompts and analyzing the gender, skin tone, and age of the generated subjects as well as the potential for NSFW content. Here's how the analysis works:

1. **Image Generation**: For each prompt, 10 images are generated using the selected model.
2. **Gender Detection**: The [BLIP caption generator](https://huggingface.co/Salesforce/blip-image-captioning-large) is used to elicit gender markers by identifying words like "man," "boy," "woman," and "girl" in the captions.
3. **Skin Tone Classification**: The [skin-tone-classifier library](https://github.com/ChenglongMa/SkinToneClassifier) is used to extract the skin tones of the generated subjects.
4. **Age Detection**: The [Faces Age Detection model](https://huggingface.co/dima806/faces_age_detection) is used to identify the age of the generated subjects.
5. **NSFW Detection**: The [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) model is used to identify whether the generated images are NSFW (not safe for work).

## Models

- Sintetico XL: a merged model with my favorite aesthetics
- Sintetico XL Prude: a SFW version that aims to remove unwanted nudity and sexual content.


''')
    with gr.Accordion("Open for More Information!", open=False):
        gr.Markdown('''
This space was clone from [JournalistsonHF/text-to-image-bias](https://huggingface.co/spaces/JournalistsonHF/text-to-image-bias).

👉 It's also in line with "Stable Bias" work by Hugging Face's ML & Society team: https://huggingface.co/spaces/society-ethics/StableBias     

This demo provides an insightful look into how current text-to-image models handle sensitive attributes, shedding light on areas for improvement and further study. 
[Here is an article](https://medium.com/@evijit/analysis-of-ai-generated-images-of-indian-people-for-colorism-and-sexism-b80ff946759f) showing how this space can be used to perform such analyses, using colorism and sexism in India as an example.          

#### Visualization

We create visual grids to represent the data:

- **Skin Tone Grids**: Skin tones are plotted as exact hex codes rather than using the Fitzpatrick scale, which can be [problematic and limiting for darker skin tones](https://arxiv.org/pdf/2309.05148).
- **Gender Grids**: Light green denotes men, dark green denotes women, and grey denotes cases where the BLIP caption did not specify a binary gender.
- **Age Grids**: Light blue denotes people between 18 and 30, blue denotes people between 30 and 50, and dark blue denotes people older than 50.
- **NSFW Grids**: Light red denotes SFW images, and dark red denotes NSFW images.

''')
    model_dropdown = gr.Dropdown(
        label="Choose a model", 
        choices=[
            "sinteticoXL",
            "sinteticoXL_Prude"
        ], 
        value=default_model
    )
    prompt = gr.Textbox(label="Enter the Prompt", value = "photo of a Brazilian scientist, high quality, good lighting")
    gallery = gr.Gallery(
        label="Generated images", 
        show_label=False, 
        elem_id="gallery", 
        columns=[5], 
        rows=[2], 
        object_fit="contain", 
        height="auto"
    )
    btn = gr.Button("Generate images", scale=0)
    with gr.Row(equal_height=True):
        skinplot = gr.Plot(label="Skin Tone")
        genplot = gr.Plot(label="Gender")
    with gr.Row(equal_height=True):
        agesplot = gr.Plot(label="Age")
        nsfwsplot = gr.Plot(label="NSFW")
    btn.click(generate_images_plots, inputs=[prompt, model_dropdown], outputs=[gallery, skinplot, genplot, agesplot, nsfwsplot])

demo.launch(debug=True)