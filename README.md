# sinteticoxl-bias

Live here: [Bias detection in SinteticoXL Models](https://huggingface.co/spaces/lucianosb/sinteticoXL-bias)

This repo is a mirror from the Space originally published on Hugging Face.

In order to enhance model transparency and ensure ethical AI practices, I made a fork of [text-to-image-bias](https://huggingface.co/spaces/JournalistsonHF/text-to-image-bias) to apply their approach to my own merged models.

In this demo, we explore the potential biases in text-to-image models by generating multiple images based on user prompts and analyzing the gender, skin tone, and age of the generated subjects as well as the potential for NSFW content. Here's how the analysis works:

1. **Image Generation**: For each prompt, 10 images are generated using the selected model.
2. **Gender Detection**: The [BLIP caption generator](https://huggingface.co/Salesforce/blip-image-captioning-large) is used to elicit gender markers by identifying words like "man," "boy," "woman," and "girl" in the captions.
3. **Skin Tone Classification**: The [skin-tone-classifier library](https://github.com/ChenglongMa/SkinToneClassifier) is used to extract the skin tones of the generated subjects.
4. **Age Detection**: The [Faces Age Detection model](https://huggingface.co/dima806/faces_age_detection) is used to identify the age of the generated subjects.
5. **NSFW Detection**: The [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) model is used to identify whether the generated images are NSFW (not safe for work).

## Available Models

- [Sintetico XL](https://civitai.com/models/275631/sintetico-xl): a merged model with my favorite aesthetics
- [Sintetico XL Prude](https://civitai.com/models/320690/sintetico-xl-prude): a SFW version that aims to remove unwanted nudity and sexual content.

---

Feel free to contribute!

