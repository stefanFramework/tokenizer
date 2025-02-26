from config import TokenizerConfiguration
from huggingface_hub import login
from pipeline.pipeline import Pipeline
from pipeline.steps import LoadTranscript, PreparePrompt, TokenizeInput, LoadModel, GenerateResponse

login(TokenizerConfiguration.HF_TOKEN)

pipeline = Pipeline([
    LoadTranscript(),
    PreparePrompt(),
    TokenizeInput(),
    LoadModel(),
    GenerateResponse(),
])

context = pipeline.run()
result = context["response"]

print(result)
