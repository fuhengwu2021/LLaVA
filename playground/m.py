from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import os

here = os.path.dirname(os.path.realpath(__file__))

'''
model_path = "liuhaotian/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)'''


model_path = "liuhaotian/llava-v1.5-13b"
model_path = "openai/clip-vit-large-patch14-336"
model_path = "liuhaotian/llava-v1.5-7b"
prompt = "What are written on the image and what is the bouding box position?"
image_file = f"{here}/button.jpg"
image_file = f"{here}/art-01107.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "temperature": 0.6,
    "top_p": 0.75,
    "num_beams": 1,
    "max_new_tokens": 1024,
    "load_in_8bit": True,
    "sep": ",",
})()

if __name__ == "__main__":
    eval_model(args)
