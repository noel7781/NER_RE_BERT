import config
import torch
import transformers

class NERDataset:
    def __init__(self, texts, tags):
        self.texts = texts
        self.tags = tags

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        tag = self.tags[index]

        ids = []
        target_tag = []
        acc = 1
        token_starts = []
        for i, s in enumerate(text):
            inputs = config.TOKENIZER.encode(
                    str(s),
                    add_special_tokens=False,
                    )
            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend([tag[i]] * input_len)
            token_starts.append(acc)
            acc += input_len

        ids = ids[:config.MAX_LEN - 2]
        target_tag = target_tag[:config.MAX_LEN - 2]

        ids = [101] + ids + [102]
        target_tag = [-100] + target_tag + [-100]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = config.MAX_LEN - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([-100] * padding_len)
        here = len(token_starts)
        if here >= config.MAX_LEN:
            token_starts = token_starts[:config.MAX_LEN]
        token_starts = token_starts + ([0] * (config.MAX_LEN - here))

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
            "token_starts": torch.tensor(token_starts, dtype=torch.long),
        }
