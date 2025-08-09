import argparse
from vdb import VectorDB
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

TEMPLATE = """You are a concise assistant for the Glasgow University Fencing Club.
Use only the context to answer. Do NOT repeat the question or context. Answer in 2-4 sentences.

Context:
{context}

Question: {question}
Answer:
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a vector DB")
    # Input
    parser.add_argument(
        "--question",
        help="The question that needs to be answered",
        default="what is flash attention?",
    )
    # Input
    parser.add_argument(
        "--vdb",
        type=str,
        default="vdb.npz",
        help="The path to read the vector DB",
    )
    
    args = parser.parse_args()
    m = VectorDB(args.vdb)
    context = m.query(args.question)
    prompt = TEMPLATE.format(context=context, question=args.question)
    #model, tokenizer = load("mlx-community/NeuralBeagle14-7B-4bit-mlx")
    model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    
    out = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=250,
        verbose=True,
    )
    print(out)
