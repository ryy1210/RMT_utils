"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union

alpaca_template = {
    "description": "Template used by Alpaca-LoRA.",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:"    
}

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name or template_name == 'alpaca':
            self.template = alpaca_template.copy()
        else:
            # 修正: 未初期化templateで後から落ちる代わりに、JSONを読み検証する。
            path = template_name if osp.isfile(template_name) else osp.join(osp.dirname(__file__), 'templates', template_name + '.json')
            with open(path, encoding='utf-8') as stream:
                self.template = json.load(stream)
            required = {'prompt_input', 'prompt_no_input', 'response_split'}
            if not required.issubset(self.template):
                raise ValueError('Invalid prompt template')
            self.template.setdefault('description', template_name)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        # 修正: 区切りなしや応答中の再出現でIndexError/切捨てにならない。
        return output.split(self.template["response_split"], 1)[-1].strip()


class ZeroPrompter(object):
    __slots__ = ("_verbose")

    def __init__(self, verbose: bool = False):
        self._verbose = verbose
        
        if self._verbose:
            print(
                f"Without using prompt template!"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        # 修正: 空のinstructionを許し、[-1]のIndexErrorを防ぐ。
        instruction = instruction or ''
        if instruction and instruction[-1] == '.':
            instruction = instruction[:-1] + ':'
        if instruction and instruction[-1] not in ['.', ':', '?', '!']:
            instruction = instruction + ':'
        instruction += ' '

        if input:
            if input[-1] not in ['.', ':', '?', '!']:
                input = input + '.'
            res = instruction + input
        else:
            res = instruction
        if label:
            res = f"{res} {label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.strip()
