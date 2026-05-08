import sys, os

# Fix import path conflict (vllm/ repo dir vs installed package)
sys.path = [p for p in sys.path if not p == os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vllm"))

# Suppress vllm's verbose INFO logs
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

from vllm import LLM, SamplingParams

if __name__ == "__main__":
    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

    prompt = "What is the capital of France?"

    outputs = llm.generate([prompt], sampling_params)
    print(outputs[0].outputs[0].text)
