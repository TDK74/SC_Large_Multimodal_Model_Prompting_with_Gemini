import sys
import typing

import IPython
import IPython.display
import vertexai

from PIL import Image as PIL_Image
from PIL import ImageOps as PIL_ImageOps
from utils import (gemini_vision, gemini_vision_parameters, print_multimodal_prompt)
from vertexai.generative_models import (GenerationConfig, GenerativeModel, Image, Part)


## ------------------------------------------------------ ##
app = IPython.Application.instance()
app.kernel.do_shutdown(True)

## ------------------------------------------------------ ##
if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()

## ------------------------------------------------------ ##
PROJECT_ID = "moonlit-conduit-474111-i5"
LOCATION = "europe-central2"

vertexai.init(project = PROJECT_ID, location = LOCATION)

## ------------------------------------------------------ ##
def gemini(prompt, model):
    responses = model.generate_content(prompt, stream = True)

    response_text = ""

    for response in responses:
        response_text += response.text

    return response_text


def display_images(images: typing.Iterable[Image], max_width: int = 600, max_height: int = 350,
                   ) -> None:
    for image in images:
        pil_image = typing.cast(PIL_Image.Image, image._pil_image)

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        image_width, image_height = pil_image.size

        if max_width < image_width or max_height < image_height:
            pil_image = PIL_ImageOps.contain(pil_image, (max_width, max_height))

        IPython.display.display(pil_image)


def print_multimodal_prompt(contents: list):
    for content in contents:
        if isinstance(content, Image):
            display_images([content])

        elif isinstance(content, Part):
            url = get_url_from_gcs(content.file_data.file_uri)
            IPython.display.display(load_image_from_url(url))

        else:
            print(content)


def gemini_vision(contents_image, model):
    responses = model.generate_content(contents_image, stream = True)

    response_text = ""

    for response in responses:
        response_text += response.text

    return response_text


def gemini_vision_parameters(contents_image, model, config):
    responses = model.generate_content(contents = contents_image,
                                        generation_config = config,
                                        stream = True)

    response_text = ""

    for response in responses:
        response_text += response.text

    return response_text

## ------------------------------------------------------ ##
model = GenerativeModel("gemini-2.0-flash")

## ------------------------------------------------------ ##
gemini("What is a multimodal model?", model = model)

## ------------------------------------------------------ ##
prompt_1 = """ In short, what is deeplearning.ai, and what can it offer me as a Machine Learning
            Engineer? """

## ------------------------------------------------------ ##
response_1 = model.generate_content(prompt_1, stream = True)

## ------------------------------------------------------ ##
print(response_1)

## ------------------------------------------------------ ##
for response in response_1:
    print(response)

## ------------------------------------------------------ ##
response_1 = model.generate_content(prompt_1, stream = True)

## ------------------------------------------------------ ##
for response in response_1:
    print(response.text)

## ------------------------------------------------------ ##
multimodal_model = GenerativeModel("gemini-2.0-flash")

## ------------------------------------------------------ ##
image = Image.load_from_file("andrew_power_tools.png")

## ------------------------------------------------------ ##
prompt_3 = "Please describe what is in this image?"

# prompt_3 = "What are likely professions of this person?"

## ------------------------------------------------------ ##
contents_image = [image, prompt_3]

## ------------------------------------------------------ ##
print("-------Prompt--------")
print_multimodal_prompt(contents_image)

## ------------------------------------------------------ ##
gemini_vision(contents_image, model = multimodal_model)

## ------------------------------------------------------ ##
file_path = "dlai-sc-gemini-bucket/pixel8.mp4"
video_uri = f"gs://{file_path}"
video_url = f"https://storage.googleapis.com/{file_path}"

## ------------------------------------------------------ ##
IPython.display.Video(video_url, width = 450)

## ------------------------------------------------------ ##
prompt = """ Answer the following questions using the video only:
            - What is the main person's profession?
            - What are the main features of the phone highlighted?
            - Which city was this recorded in? """

## ------------------------------------------------------ ##
video = Part.from_uri(video_uri, mime_type = "video/mp4")
contents_video = [prompt, video]

## ------------------------------------------------------ ##
responses_4 = multimodal_model.generate_content(contents_video, stream = True)

## ------------------------------------------------------ ##
for response in responses_4:
    print(response.text, end = "")

## ------------------------------------------------------ ##
image_1 = Image.load_from_file("./panda.png")

## ------------------------------------------------------ ##
prompt_1 = """ Write what is happening in the following image from a unique perspective
            and do not mention names. """

## ------------------------------------------------------ ##
contents = [image_1, prompt_1]

## ------------------------------------------------------ ##
print("-------Prompt--------")
print_multimodal_prompt(contents)

## ------------------------------------------------------ ##
response_1 = multimodal_model.generate_content(contents, stream = True)

## ------------------------------------------------------ ##
for response in response_1:
    print(response.text, end = "")

## ------------------------------------------------------ ##
generation_config_1 = GenerationConfig(temperature = 0.0, top_k = 1, )

## ------------------------------------------------------ ##
response_zero_temp = gemini_vision_parameters(contents, multimodal_model, generation_config_1)

## ------------------------------------------------------ ##
print(response_zero_temp)

## ------------------------------------------------------ ##
responses_zero_temp = gemini_vision_parameters(contents, multimodal_model, generation_config_1)

print(response_zero_temp)

## ------------------------------------------------------ ##
generation_config_2 = GenerationConfig(temperature = 1, top_k = 40, )

## ------------------------------------------------------ ##
print(responses_high_temp_topk)

## ------------------------------------------------------ ##
generation_config_4 = GenerationConfig(temperature = 1, top_k = 40, top_p = 0.01, )

## ------------------------------------------------------ ##
responses_high_temp_topp = gemini_vision_parameters(contents, multimodal_model, generation_config_4)

print(responses_high_temp_topp)

## ------------------------------------------------------ ##
generation_config_5 = GenerationConfig(max_output_tokens = 10, )

## ------------------------------------------------------ ##
responses_max_output = gemini_vision_parameters(contents, multimodal_model, generation_config_5)

print(responses_max_output)

## ------------------------------------------------------ ##
generation_config_6 = GenerationConfig(stop_sequences = ["panda"])

## ------------------------------------------------------ ##
responses_stop = gemini_vision_parameters(contents, multimodal_model, generation_config_6)

print(responses_stop)
