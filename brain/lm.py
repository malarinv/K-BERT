import torch

# import sys
import numpy as np

from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model (weights)
with torch.no_grad():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def score(sentence):
    tokenize_input = tokenizer.encode(sentence)
    tensor_input = torch.tensor([tokenize_input])
    loss = model(tensor_input, labels=tensor_input)[0]
    return np.exp(loss.detach().numpy())


def score_list(sentences):
    tokenize_input = tokenizer(sentences, padding=True)["input_ids"]
    import pdb

    pdb.set_trace()
    tensor_input = torch.tensor(tokenize_input)
    import pdb

    pdb.set_trace()
    loss = model(tensor_input, labels=tensor_input)[0]
    import pdb

    pdb.set_trace()
    return np.exp(loss.detach().numpy())


if __name__ == "__main__":
    # print(
    #     score(
    #         "semantic web technologies are finally after a few years of infancy truly entering the business world to support the growing needs of computer aided information selection and processing there are already quite well defined development processes and methods in the soft ware engineering field to handle the construction of large scale and complex enterprise systems and to reuse knowledge in different soft ware domains patterns are considered to be common practise patterns can be described on different levels of abstraction but the patterns in the focus of this paper are on the software architecture level in this paper we present a definition of the notion ontology application pattern as a special form of soft ware architecture patterns describing an ontology based system we also show how such patterns as well as the description of the pattern instantiations can be described using a modified architecture description language"
    #     )
    # )
    print(
        score_list(
            [
                "semantic web technologies are finally after a few years of infancy truly entering the business world to support the growing needs of computer aided information selection and processing there are already quite well defined development processes and methods in the soft ware engineering field to handle the construction of large scale and complex enterprise systems and to reuse knowledge in different soft ware domains patterns are considered to be common practise patterns can be described on different levels of abstraction but the patterns in the focus of this paper are on the software architecture level in this paper we present a definition of the notion ontology application pattern as a special form of soft ware architecture patterns describing an ontology based system we also show how such patterns as well as the description of the pattern instantiations can be described using a modified architecture description language",
                "semantic web technologies are finally after a few years of infancy truly entering the business world to support the growing needs of computer aided information selection and processing there are already quite well defined development processes and methods in the soft ware engineering field to handle the construction of large scale and complex enterprise systems and to reuse knowledge in different soft ware domains patterns are considered to be common practise patterns can be described on different levels of abstraction but the patterns in the focus of this paper are on the soft ware architecture level in this paper we present a definition of the notion ontology application pattern as a special form of soft ware architecture patterns describing an ontology based system we also show how such patterns as well as the description of the pattern instantiations can be described using a modified architecture description language",
            ]
        )
    )
