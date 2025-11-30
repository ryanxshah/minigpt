import torch


with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))

stoi = {char:i for i, char in enumerate(chars)}
itos = {i:char for i, char in enumerate(chars)}

def encode(string):
    return [stoi[char] for char in string]

def decode(list):
    return "".join([itos[i] for i in list])