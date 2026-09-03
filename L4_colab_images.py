import sys
import typing
import IPython
import IPython.display
import vertexai

from PIL import Image as PIL_Image
from PIL import ImageOps as PIL_ImageOps
from utils import gemini_vision, print_multimodal_prompt
from vertexai.generative_models import GenerativeModel, Image
from vertexai.preview.generative_models import Image, Part


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
def display_images(
                  images: typing.Iterable[Image],
                  max_width: int = 600,
                  max_height: int = 350,
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

        print("\n")


def gemini_vision(contents, model):
    responses = model.generate_content(contents, stream = True)

    response_text = ""

    for response in responses:
        response_text += response.text

    return response_text

## ------------------------------------------------------ ##
multimodal_model = GenerativeModel("gemini-2.0-flash")

## ------------------------------------------------------ ##
fruit = Image.load_from_file("./bowl-fruits.jpg")

## ------------------------------------------------------ ##
prices = Image.load_from_file("./price-list-fruits.jpg")

## ------------------------------------------------------ ##
images = [fruit, prices]

## ------------------------------------------------------ ##
print("-------images--------")
print_multimodal_prompt(images)

## ------------------------------------------------------ ##
instruction_1 = """
                I want to make a fruit salad with three bananas, two apples, \
                one kiwi, and one orange. This is an image of my bowl \
                of fruits:
                """

## ------------------------------------------------------ ##
instruction_2 = "This is the price list for fruits at \
                my supermarket:"

## ------------------------------------------------------ ##
question = """
          Please answer these questions:
          - Describe which fruits and how many I have in my fruit bowl on \
          the image?
          - Given the fruits in my bowl on the image and the fruit salad \
          recipe, what am I missing?
          - Given the fruits I still need to buy, what \
          would be the prices and total cost for these fruits?
          """

## ------------------------------------------------------ ##
contents = [
            instruction_1,
            fruit,
            instruction_2,
            prices,
            question,
          ]

## ------------------------------------------------------ ##
contents = [
            instruction_1,
            fruit,
            instruction_2,
            prices,
            question,
          ]

## ------------------------------------------------------ ##
print("-------Prompt--------")
print_multimodal_prompt(contents)

## ------------------------------------------------------ ##
print("\n-------Response--------\n")

response = gemini_vision(contents, multimodal_model)

print(response, end = "")

## ------------------------------------------------------ ##
furniture_images_uri = [
                        "./chair-1.jpg",
                        "./chair-2.jpg",
                        "./chair-3.jpg",
                        "./chair-4.jpg",
                      ]

## ------------------------------------------------------ ##
room_image = Image.load_from_file("./room.jpg")

## ------------------------------------------------------ ##
furniture_images = [Image.load_from_file(uri)
                    for uri in furniture_images_uri]

## ------------------------------------------------------ ##
images_2 = [room_image]

## ------------------------------------------------------ ##
images_2.extend(furniture_images)

## ------------------------------------------------------ ##
print("-------images--------")
print_multimodal_prompt(images_2)

## ------------------------------------------------------ ##
recommendation_content = [
                        "You are an interior designer.",
                        "Consider the following chairs:",
                        "chair 1:",
                        furniture_images[0],
                        "chair 2:",
                        furniture_images[1],
                        "chair 3:",
                        furniture_images[2],
                        "chair 4:",
                        furniture_images[3],
                        "room:",
                        room_image,
                        "For each chair, \
                        explain whether it would be appropriate for the \
                        style of the room:",
                        ]

## ------------------------------------------------------ ##
print("-------Prompt--------")
print_multimodal_prompt(recommendation_content)

## ------------------------------------------------------ ##
# multimodal_model = GenerativeModel("gemini-1.0-pro-vision-001")
multimodal_model = GenerativeModel("gemini-2.5-flash")

print("\n-------Response--------\n")

response = gemini_vision(recommendation_content, multimodal_model)

print(response, end = "")

## ------------------------------------------------------ ##
multimodal_model = GenerativeModel("gemini-2.0-flash")

## ------------------------------------------------------ ##
receipt_images_uri = [
                    './breakfast.jpg',
                    './lunch.jpg',
                    './diner.jpg',
                    './meal-others.jpg',
                    ]

## ------------------------------------------------------ ##
receipt_images = [Image.load_from_file(uri)
                  for uri in receipt_images_uri]

## ------------------------------------------------------ ##
with open("travel-policy.txt", "r") as file:
    policy = file.read()

## ------------------------------------------------------ ##
INSTRUCTION = "Never make up facts, and if you are not 100% sure, \
              be transparent in stating when you are not sure, or do not \
              have enough information to answer certain questions or \
              fulfill certain requests."

## ------------------------------------------------------ ##
ROLE = "You are an HR professional and an expert in travel expenses."

## ------------------------------------------------------ ##
ASSIGNMENT = """
            You are reviewing travel expenses for a business trip.
            Please complete the following tasks:
            1. Itemize everything on the receipts, including tax and \
            total.  This means identifying the cost of individual \
            items that add up to the total cost before tax, as well \
            as the tax ,such as sales tax, as well as tip.
            2. What is the total sales tax paid?  In some cases, \
            the total sales tax may be a sum of more than one line \
            item of the receipt.
            3. For this particular receipt, the employee who is \
            adding this business expense purchased the meal with \
            a group. The employee only ordered the KFC Bowl. Please \
            provide the cost of the employee's order only.  Include \
            both the cost before tax, and also estimate the tax \
            that is applied to this employee's order.  To do this,\
            calculate the fraction of the employee's pre-tax order\
            divided by the total pre-tax cost.  This fraction can be \
            applied to the total sales tax that you calculated earlier.
            4.  Please calculate the amount spent by others, which \
            are all the other line items on the receipt.  Please \
            provide this sum before tax, and if possible, apply the \
            tax for the total cost.
            5. Check the expenses against company policy and flag \
            if there are issues.
            """


## ------------------------------------------------------ ##
receipt_content = [
                  INSTRUCTION,
                  ROLE,
                  "Answer the questions based on the following receipts:"
                  "breakfast:",
                  receipt_images[0],
                  "lunch:",
                  receipt_images[1],
                  "diner",
                  receipt_images[2],
                  "meal-others",
                  receipt_images[3],
                  ASSIGNMENT,
                  policy,
                  ]

## ------------------------------------------------------ ##
print_multimodal_prompt(receipt_content)

## ------------------------------------------------------ ##
print("\n-------Response--------\n")

response = gemini_vision(receipt_content, multimodal_model)

print(response, end = "")
