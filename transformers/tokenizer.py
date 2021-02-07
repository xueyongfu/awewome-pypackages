from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('vocab')
tokens = tokenizer.tokenize('市民来电反映：其6月4日20:00经过该处，发现路口的人行道是行有2个摊位，在卖小吃，'
                            '影响到通行。诉求：取缔1998年。（信息保密，无需回复）')
print(tokens)