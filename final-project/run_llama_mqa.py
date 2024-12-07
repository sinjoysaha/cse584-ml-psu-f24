from transformers import pipeline
import torch
import os, json
from datasets import load_dataset
from tqdm import tqdm
import datetime

print(torch.cuda.device_count())
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print(torch.cuda.device_count())


model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # "meta-llama/Meta-Llama-3.1-70B"
math_dataset = load_dataset("hendrycks/competition_math")

pipe = pipeline(
    "text-generation",
    model=model_name,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

NUM_SAMPLES = 5000
tlist = math_dataset["train"][0:NUM_SAMPLES]["type"]
qlist = math_dataset["train"][0:NUM_SAMPLES]["problem"]
alist = math_dataset["train"][0:NUM_SAMPLES]["solution"]

output_file = (
    f"""results/results_{datetime.datetime.now().strftime("%Y%m%d-%H%M")}.json"""
)
error_file = f"""results/error_{datetime.datetime.now().strftime("%Y%m%d-%H%M")}.json"""
print(f"{output_file=}")
print(f"{error_file=}")


def prompt_for_falsemathqgen(valid_question, solution):
    return f"Question: {valid_question}\nSolution: {solution}\nModify the problem into a false math problem that is unsolvable. Your answer should be exactly in this format.\nFalse Math Problem: []  \nSimple Explanation: []"


def process_mathqresp(resp):
    false_problem = ""
    simple_explanation = ""
    try:
        false_problem = (
            resp.split("False Math Problem:")[1].split("Simple Explanation:")[0].strip()
        )
        simple_explanation = resp.split("Simple Explanation:")[1].strip()
        # print("False Q:", false_problem)
        # print("Exp:", simple_explanation)
    except Exception as e:
        print("Error occured.", e, sep="\n")

    return false_problem, simple_explanation


def generate(prompt):
    messages = [
        {"role": "user", "content": prompt},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=512,
    )
    generation = outputs[0]["generated_text"][-1]
    resp = generation["content"]
    return (generation, resp)


main_list = []
error_list = []

for idx, tup in tqdm(enumerate(zip(tlist, qlist, alist))):
    try:
        t, q, a = tup
        temp_dict = {
            "original_question": q,
            "original_solution": a,
            # "raw_generation": rawgen,
            "generations": [],
        }
        for i in range(3):
            rawgen, resp = generate(
                prompt_for_falsemathqgen(valid_question=q, solution=a)
            )
            false_problem, simple_explanation = process_mathqresp(resp)
            temp_dict["generations"].append(
                {
                    "resp": resp,
                    "false_problem": false_problem,
                    "simple_explanation": simple_explanation,
                }
            )
            print(temp_dict)

            if false_problem == "" and simple_explanation == "":
                print("Error: ", idx, q, resp, sep="\n")
                error_list.append(temp_dict)
                with open(error_file, "w") as f:
                    json.dump(error_list, f, indent=4)
        # else:
        main_list.append(temp_dict)
        with open(output_file, "w") as f:
            json.dump(main_list, f, indent=4)
    except Exception as e:
        print(e)
