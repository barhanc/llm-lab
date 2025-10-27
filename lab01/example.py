import argparse
import torch

from gpt import GPTLanguageModel
from gru import GRULanguageModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, choices=("gru", "gpt"))
    parser.add_argument("-p", "--prompt", type=str, default="\n")
    args = parser.parse_args()
    model_name = args.model

    # fmt:off
    vocab = ['\n',' ','!',"'",'(',')','*',',','-','.','/','0','1','2','3','4',
            '5','6','7','8','9',':',';','?','A','B','C','D','E','F','G','H',
            'I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X',
            'Y','Z','[',']','a','b','c','d','e','f','g','h','i','j','k','l',
            'm','n','o','p','q','r','s','t','u','v','w','x','y','z','«','»',
            'Ó','à','â','ä','æ','ç','è','é','ê','ó','ô','ü','Ą','ą','Ć','ć',
            'Ę','ę','Ł','ł','Ń','ń','Ś','ś','Ź','ź','Ż','ż','–','—','’','”',
            '„','…']
    # fmt:on
    vocab_size = len(vocab)

    itos = {i: ch for i, ch in enumerate(vocab)}
    stoi = {ch: i for i, ch in enumerate(vocab)}

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device.upper()}")

    if model_name == "gpt":
        model = GPTLanguageModel(
            vocab_size=vocab_size,
            block_size=256,
            num_layers=6,
            dim_model=384,
            num_heads=6,
            dropout_p=0.2,
        ).to(device)
        gpt_chkpt = torch.load("models/gpt_final.pt", map_location=device)
        model.load_state_dict(gpt_chkpt["model"])

    elif model_name == "gru":
        model = GRULanguageModel(
            vocab_size=vocab_size,
            block_size=256,
            num_layers=6,
            dim_model=512,
            dropout_p=0.2,
        ).to(device)
        gru_chkpt = torch.load("models/gru_final.pt", map_location=device)
        model.load_state_dict(gru_chkpt["model"])

    else:
        raise ValueError

    prompt = args.prompt
    print(prompt, end="")
    prompt = [stoi[ch] for ch in prompt]

    ctx = torch.tensor(prompt, dtype=torch.long).to(device)
    for tok in model.generate(ctx):
        print(itos[tok], end="")
