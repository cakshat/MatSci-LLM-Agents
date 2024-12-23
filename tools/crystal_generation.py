import os
import re
from pymatgen.core import Composition
from contextlib import nullcontext
import torch

from crystallm import (
    CIFTokenizer,
    GPT,
    GPTConfig,
    extract_space_group_symbol,
    replace_symmetry_operators,
    remove_atom_props_block,
    get_atomic_props_block_for_formula
)
from typing import Optional, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from openai import OpenAI
import json
from datetime import datetime

OPENAI_API = os.getenv("OPENAI_API")

def get_prompt(comp, sg=None):
    # NOTE: we have to use comp.formula, so that the elements are sorted by electronegativity,
    #  which is what the model saw in training; comp.formula looks something like 'Zn1 Cu1 Te1 Se1',
    #  so we have to strip the spaces
    comp_str = comp.formula.replace(" ", "")
    if sg is not None:
        # construct an input string with the space group
        block = get_atomic_props_block_for_formula(comp_str)
        cif_str = f"data_{comp_str}\n{block}\n_symmetry_space_group_name_H-M {sg}\n"
        # strip out any leading or trailing spaces from the prompt
        cif_str = re.sub(r"^[ \t]+|[ \t]+$", "", cif_str, flags=re.MULTILINE)
        return cif_str
    else:
        return f"data_{comp_str}\n"

def make_prompt_file(composition, sg=None, prompt_fname="my_prompt.txt"):
    comp = Composition(composition)
    prompt = get_prompt(comp, sg)
    return prompt

class SampleDefaults:
    out_dir: str = "crystallm/crystallm_v1_small"  # the path to the directory containing the trained model
    start: str = "FILE:my_prompt.txt"  # the prompt; can also specify a file, use as: "FILE:prompt.txt"
    num_samples: int = 2  # number of samples to draw
    max_new_tokens: int = 3000  # number of tokens generated in each sample
    temperature: float = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k: int = 10  # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed: int = 1337
    device: str = "cpu"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype: str = "bfloat16"  # 'float32' or 'bfloat16' or 'float16'
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster
    target: str = "crystallm_cifs/generated_cifs"  # where the generated content will be sent; can also be 'file'

def generate_cif_files(C):
    torch.manual_seed(C.seed)
    torch.cuda.manual_seed(C.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in C.device else "cpu"  # for later use in torch.autocast
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[C.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    tokenizer = CIFTokenizer()
    encode = tokenizer.encode
    decode = tokenizer.decode

    ckpt_path = os.path.join(C.out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=C.device, weights_only=True)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(C.device)
    if C.compile:
        model = torch.compile(model)

    prompt = C.start
    if prompt.startswith("FILE:"):
        with open(prompt[5:], "r", encoding="utf-8") as f:
            prompt = f.read()
    start_ids = encode(tokenizer.tokenize_cif(prompt))
    x = torch.tensor(start_ids, dtype=torch.long, device=C.device)[None, ...]

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(C.num_samples):
                y = model.generate(x, C.max_new_tokens, temperature=C.temperature, top_k=C.top_k)

                generated = decode(y[0].tolist())

                if C.target == "console":
                    print(generated)
                    print('---------------')
                elif C.target == "file":
                    fname = f"sample_{k+1}.cif"
                    print(f"writing generated content to {fname} ...")
                    with open(fname, "wt") as f:
                        f.write(generated)
                else:
                    fname = f"{C.target}/sample_{k+1}.cif"
                    print(f"writing generated content to {fname} ...")
                    with open(fname, "wt") as f:
                        f.write(generated)

def postprocess(cif: str, fname: str) -> str:
    try:
        # replace the symmetry operators with the correct operators
        space_group_symbol = extract_space_group_symbol(cif)
        if space_group_symbol is not None and space_group_symbol != "P 1":
            cif = replace_symmetry_operators(cif, space_group_symbol)

        # remove atom props
        cif = remove_atom_props_block(cif)
    except Exception as e:
        cif = "# WARNING: CrystaLLM could not post-process this file properly!\n" + cif
        print(f"error post-processing CIF file '{fname}': {e}. Using original file.")

    return cif

def post_process_cif_files(input_path="crystallm_cifs/generated_cifs", output_path="crystallm_cifs/processed_cifs"):
    post_processed_file_paths = []
    post_processed_cifs = []
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(input_path):
        if filename.endswith(".cif"):
            file_path = os.path.join(input_path, filename)
            with open(file_path, "r") as file:
                cif_str = file.read()
                processed_cif = postprocess(cif_str, filename)

            output_file_path = os.path.join(output_path, filename)
            with open(output_file_path, "w") as file:
                file.write(processed_cif)
            print(f"processed: {filename}")
            post_processed_file_paths.append(output_file_path)
            post_processed_cifs.append(processed_cif)
    return post_processed_file_paths, post_processed_cifs

def generate_summary(cif: str) -> str:
    """
    Converts a cif file into simple text summary for the user.
    """
    results_str = json.dumps(cif)
    client = OpenAI(api_key=OPENAI_API)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant that generates concise, informative summaries."
            },
            {
                "role": "user", 
                "content": f"""Please generate a concise, informative summary of the cif file.
                Highlight key insights, patterns, and most significant findings.
                Keep the summary under 300 words and focus on the most important information.
                
                Results:
                {results_str}
                
                Summary:"""
            }
        ],
        max_tokens=350
    )
    return response.choices[0].message.content

class CrystaLLMInput(BaseModel):
    composition: str = Field(
        "NaCl", 
        description="Chemical formula of the compound whose crystal structure is to be generated"
    )
    sg: str = Field(
        None, 
        description="Space group for the crystal structure."
    )
    model_config = {
        "extra": "forbid",
        "arbitrary_types_allowed": True
    }

class CrystaLLMTool(BaseTool):
    name: str = "CrystaLLM"
    description: str = """
    CrystaLLM is a Transformer-based Large Language Model of the CIF (Crystallographic Information File) format.
    The model can be used to generate crystal structures for a given composition and, optionally, space group.
    """
    args_schema: Type[BaseModel] = CrystaLLMInput

    def _run(
        self, 
        composition: str = "NaCl", 
        sg: Optional[str] = None
    ) -> str:
        """Run the CrystaLLM tool."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generated_folder_name = f"crystallm_cifs/{composition}_{timestamp}/generated_cifs"
        processed_folder_name = f"crystallm_cifs/{composition}_{timestamp}/processed_cifs"
        os.makedirs(generated_folder_name)
        os.makedirs(processed_folder_name)
        prompt = make_prompt_file(composition=composition, sg=sg)
        C = SampleDefaults
        C.start = prompt
        C.target = generated_folder_name
        generate_cif_files(C)
        post_processed_files, post_processed_cifs = post_process_cif_files(
            input_path=C.target, 
            output_path=processed_folder_name
        )
        return generate_summary(post_processed_cifs[0])