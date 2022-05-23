# %%
from flask import Flask, request
import random
import warnings
import torch
from parrot import Parrot

# %%
# pip install git+https: // github.com/PrithivirajDamodaran/Parrot_Paraphraser.git

# %%
warnings.filterwarnings("ignore")


def random_state(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


random_state(1234)
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

# def show_blog(text):
#   phrases = [text]
#   for phrase in phrases:
#    print("-"*100)
#    print("Input_phrase: ", phrase)
#    print("-"*100)
#    para_phrases = parrot.augment(input_phrase=phrase, use_gpu=False)
#    return(para_phrases[1][0])
#  for para_phrase in para_phrases:


# %%
# print(para_phrases[1][0])


def show_blog(text):
    phrases = [text]
    for phrase in phrases:
        print("-"*100)
        print("Input_phrase: ", phrase)
        print("-"*100)
        para_phrases = parrot.augment(input_phrase=phrase, use_gpu=False)
        n = len(para_phrases)
        m = random.randrange(0, n)
        return(para_phrases[m][0])


# %%


app = Flask(__name__)


@app.route('/')
def hello_world():
    return "Hello World"


@app.route('/para', methods=['GET'])
def result():
    args = request.args
    return show_blog(args.get("text"))


if __name__ == '__main__':
    app.run()
