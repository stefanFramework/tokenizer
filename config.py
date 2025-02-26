import os
from os.path import join
from dotenv import load_dotenv

load_dotenv('.env')


class TokenizerConfiguration:
    HF_TOKEN = os.getenv('HF_TOKEN')
    TEXT_MODEL = os.getenv('TEXT_MODEL', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
    TRANSCRIPT_FILE_NAME = join('./assets', os.getenv('TRANSCRIPT_FILE_NAME'))

    MAX_AMOUNT_OF_TOKENS = 2000

