from transformers import AutoTokenizer

delta_path='liuhaotian/LLaVA-13b-delta-v0'
delta_path='liuhaotian/LLaVA-7b-delta-v0'
delta_path='liuhaotian/LLaVA-13b-delta-v0-science_qa'




if __name__ == '__main__':
    delta_tokenizer = AutoTokenizer.from_pretrained(delta_path)

    print(delta_tokenizer)
