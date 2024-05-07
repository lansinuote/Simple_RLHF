import torch


class TokenizerUtil:

    def __init__(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b')
        tokenizer.bos_token = '<s>'

        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

    def encode(self, sent, max_length=512):
        input_ids = self.tokenizer.encode(sent, add_special_tokens=False)

        input_ids = [self.bos_token_id
                     ] + input_ids[:max_length - 2] + [self.eos_token_id]

        input_ids = input_ids + [self.pad_token_id
                                 ] * (max_length - len(input_ids))

        input_ids = torch.LongTensor(input_ids)
        attention_mask = (input_ids != self.pad_token_id).long()

        return input_ids, attention_mask

    def decode(self, input_ids):
        input_ids = input_ids.tolist()

        if self.eos_token_id in input_ids:
            end = input_ids.index(self.eos_token_id) + 1
            input_ids = input_ids[:end]

        return self.tokenizer.decode(input_ids)

    def pad_to_left(self, input_ids):
        input_ids = input_ids.tolist()
        end = input_ids.index(self.eos_token_id)
        #替换eos为pad
        input_ids[end] = self.pad_token_id
        input_ids = input_ids[end:] + input_ids[:end]

        input_ids = torch.LongTensor(input_ids)
        attention_mask = (input_ids != self.pad_token_id).long()

        return input_ids, attention_mask