import argparse
from vdb import VectorDB
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a chatbot that answers questions about fencing and the Glasgow University Fencing Club.
You must ONLY answer questions about fencing, fencers or the Glasgow University Fencing Club.

Use only the provided context when answering relevant questions.
Do NOT repeat the question or context.
Keep answers to one or two short paragraphs.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
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
    model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-8bit")
    
    out = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=350,
        verbose=True,
    )