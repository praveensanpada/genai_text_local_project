from llama_cpp import Llama
import os

# Make sure this matches your downloaded file
MODEL_PATH = "./models/llama2/llama-2-7b-chat.Q4_K_M.gguf"

# Initialize the model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=os.cpu_count() or 4,
    use_mlock=True,
    verbose=True,
)

# Prompt the user
user_prompt = input("🧠 Enter your prompt:\n> ")

# Run inference
print("\n⏳ Generating response...\n")
response = llm(user_prompt, max_tokens=200)
generated = response["choices"][0]["text"]

# Output
print("💬 Generated Text:\n")
print(generated.strip())
