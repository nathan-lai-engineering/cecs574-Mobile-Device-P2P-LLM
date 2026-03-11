import shutil
import onnx
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from transformers.file_utils import default_cache_path
import os
import torch
import transformers
from system_pipeline.onnx_backend.onnx import *
from system_pipeline.model_split import model_split
from system_pipeline.model_split.utils import model_sharding, max_split_size, InferenceBuffer, get_module_flops, \
    create_model_alloc
import numpy as np
import random
import re
import psutil
import gc
import time
from tqdm.auto import tqdm
import json
import zipfile

torch.fx.wrap('len')

# Patch: add LLaMA to transformers fx supported models so symbolic_trace works
try:
    import transformers.utils.fx as _tfx
    if "LlamaForCausalLM" not in _tfx._SUPPORTED_MODELS:
        _tfx._SUPPORTED_MODELS = _tfx._SUPPORTED_MODELS + ("LlamaForCausalLM",)
except Exception:
    pass


def _patch_llama_for_fx_tracing():
    """
    Replace LlamaAttention.forward with a version that:
    - Correctly uses num_key_value_heads for k/v projections (GQA support
      missing in transformers 4.33.x for TinyLlama/LLaMA-2)
    - Removes dynamic tensor size validation checks that break torch.fx tracing
    """
    try:
        import math
        import torch
        import torch.nn.functional as F
        import transformers.models.llama.modeling_llama as _llama_mod

        _apply_rotary_pos_emb = _llama_mod.apply_rotary_pos_emb

        # repeat_kv was added with GQA support — define inline as fallback
        _repeat_kv = getattr(_llama_mod, 'repeat_kv', None)
        if _repeat_kv is None:
            def _repeat_kv(hidden_states, n_rep):
                if n_rep == 1:
                    return hidden_states
                bs, num_kv, slen, head_dim = hidden_states.shape
                return (hidden_states[:, :, None, :, :]
                        .expand(bs, num_kv, n_rep, slen, head_dim)
                        .reshape(bs, num_kv * n_rep, slen, head_dim))

        def patched_forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        ):
            bsz, q_len, _ = hidden_states.size()

            # Derive num_key_value_heads from projection weight shape —
            # transformers 4.33.x doesn't store this as an instance attribute.
            num_kv_heads = self.k_proj.out_features // self.head_dim
            num_kv_groups = self.num_heads // num_kv_heads

            query_states = self.q_proj(hidden_states)
            key_states   = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states   = key_states.view(  bsz, q_len, num_kv_heads,   self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_kv_heads,   self.head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]

            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            if past_key_value is not None:
                key_states   = torch.cat([past_key_value[0], key_states],   dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states) if use_cache else None

            # Expand k/v heads to match q heads (GQA → MHA expansion)
            key_states   = _repeat_kv(key_states,   num_kv_groups)
            value_states = _repeat_kv(value_states, num_kv_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output  = torch.matmul(attn_weights, value_states)
            attn_output  = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
            attn_output  = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value

        cls = getattr(_llama_mod, 'LlamaAttention', None)
        if cls is not None:
            cls.forward = patched_forward
            print("[LLaMA fx patch] Patched LlamaAttention.forward for torch.fx tracing")
    except Exception as e:
        print(f"[LLaMA fx patch] Warning: {e}")


_patch_llama_for_fx_tracing()


def _patch_llama_rotary_for_fx_tracing():
    """
    LlamaRotaryEmbedding.forward has `if seq_len > self.max_seq_len_cached:`
    which fails fx tracing because seq_len is a proxy. The check just lazily
    extends the cos/sin cache — safe to skip since max_position_embeddings
    (default 2048) covers all inference lengths we'll use.
    """
    try:
        import transformers.models.llama.modeling_llama as _llama_mod
        cls = getattr(_llama_mod, 'LlamaRotaryEmbedding', None)
        if cls is None:
            return

        def patched_forward(self, x, seq_len=None):
            return (
                self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
                self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            )

        cls.forward = patched_forward
        print("[LLaMA fx patch] Patched LlamaRotaryEmbedding.forward for torch.fx tracing")
    except Exception as e:
        print(f"[LLaMA fx patch] Warning (rotary): {e}")


_patch_llama_rotary_for_fx_tracing()

available_models = {
    "llama-2-7b": ["/app/llama-2-7b", "sentencepiece_tokenizer"],
    "tinyllama": ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "sentencepiece_tokenizer"],
    "bloom560m": ["bigscience/bloom-560m", "huggingface_tokenizer"],
    "bloom1b1": ["bigscience/bloom-1b1", "huggingface_tokenizer"],
    "bloom1b7": ["szzzzz/chatbot_bloom_1b7", "huggingface_tokenizer"],
    "bloom3b": ["bigscience/bloom-3b", "huggingface_tokenizer"],
    "bloom7b1": ["bigscience/bloom-7b1", "huggingface_tokenizer"],
    "vicuna7b": ["lmsys/vicuna-7b-v1.3", "sentencepiece_tokenizer"],
    "vicuna13b": ["lmsys/vicuna-13b-v1.3", "sentencepiece_tokenizer"],
    "gpt-j6b": ["EleutherAI/gpt-j-6b", "huggingface_tokenizer"],
    "opt350m": ["facebook/opt-350m", "huggingface_tokenizer"],
    "opt1b3": ["facebook/opt-1.3b", "huggingface_tokenizer"],
    "opt125m": ["facebook/opt-125m", "huggingface_tokenizer"],
    "gptq-vicuna7b-8bit": ["TheBloke/vicuna-7B-v1.3-GPTQ", "huggingface_tokenizer"]
}


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


class ModelCard:
    def __init__(self, model_name,
                 quantization_option=True,
                 transformer_model_option=True,
                 load_balancing_option=False,
                 residual_connection=True,
                 task_type="Generation",
                 split_size=None):
        self.model_name = model_name  # model name can be the name in available_models, if not, it's required to enter url for retreiving models.
        self.model = None
        self.tokenizer = None
        self.quantization_option = quantization_option
        self.transformer_model_option = transformer_model_option
        self.onnx_module_to_split_path = ""
        self.onnx_module_to_send_path = ""
        self.task_type = task_type
        self.test_input = None
        self.residual_connection = residual_connection
        self.sequential_dependency_map = None
        self.residual_dependency_map = None
        self.module_flop_map = {}
        self.max_flop_module_index_val = [-1, -1]
        self.split_size = split_size
        self.tokenizer = None
        self.load_balancing_option = load_balancing_option
        self.device_module_arrangement = None

    def retreive_model_name(self):
        for key, val in available_models.items():
            if val == self.model_name:
                return key

    def load_model_and_tokenizer(self):
        """
        Loads the model and tokenizer based on the provided model name or Hugging Face model repository URL.
        """
        if self.model_name in available_models:
            model_to_load = available_models[self.model_name][0]
        else:
            model_to_load = self.model_name
        if self.task_type == "Generation":
            # TODO: Auto-GPTQ only works on linux and windows -- Junchen Zhao 2/23/2024
            if model_to_load.startswith("gptq"):
                if model_to_load.endswith("vicuna-8bit"):
                    pass
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_to_load, low_cpu_mem_usage=True)
        elif self.task_type == "Classification":
            # Currently, we support only binary classification for running experiments.
            id2label = {0: "NEGATIVE", 1: "POSITIVE"}
            label2id = {"NEGATIVE": 0, "POSITIVE": 1}
            self.model = AutoModelForSequenceClassification.from_pretrained(model_to_load, num_labels=2,
                                                                            id2label=id2label,
                                                                            label2id=label2id,
                                                                            low_cpu_mem_usage=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_to_load)
        print(f"Model and tokenizer for '{model_to_load}' loaded successfully.\n")
        return self.model, self.tokenizer

    def get_project_directory(self):
        # util/model_card.py lives one level below the project root
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def is_file_in_directory(self, file_name, directory_path):
        # Get absolute path of file and directory
        full_path = os.path.join(directory_path, file_name)
        return os.path.exists(full_path)

    def prepare_model_split(self):
        # Setup seed
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        # load model to cpu
        model, tokenizer = self.load_model_and_tokenizer()
        self.tokenizer = tokenizer

        if self.split_size:
            split_size = self.split_size
        else:
            # split model based on max number model split available
            split_size = max_split_size(model, transformer_model_option=self.transformer_model_option)

        sequential_dependency_map = None
        residual_dependency_map = None
        if self.residual_connection:
            modules, sequential_dependency_map, residual_dependency_map, checked_split_size = model_sharding(model,
                                                                                                             split_size,
                                                                                                             transformer_model_option=self.transformer_model_option,
                                                                                                             residual_connection=self.residual_connection,
                                                                                                             debug_mod=True)
            self.module_flop_map = get_module_flops(modules, tokenizer,
                                                    sequential_dependency_map, residual_dependency_map)
            self.sequential_dependency_map = sequential_dependency_map
            self.residual_dependency_map = residual_dependency_map
        else:
            modules, checked_split_size = model_sharding(model,
                                                         split_size,
                                                         transformer_model_option=self.transformer_model_option,
                                                         residual_connection=self.residual_connection,
                                                         debug_mod=False)
            self.module_flop_map = get_module_flops(modules, tokenizer)

        self.split_size = checked_split_size

        for module_index, val in self.module_flop_map.items():
            if val == sorted([val for val in self.module_flop_map.values()])[-1]:
                continue
            if val > self.max_flop_module_index_val[1]:
                self.max_flop_module_index_val[0], self.max_flop_module_index_val[1] = module_index, val
            else:
                continue

        # Fallback: if all modules share the same flop count the loop above skips
        # everything and leaves the index at -1.  Use module 0 in that case.
        if self.max_flop_module_index_val[0] == -1 and self.module_flop_map:
            first_key = next(iter(self.module_flop_map))
            self.max_flop_module_index_val = [first_key, self.module_flop_map[first_key]]

        # create input for model export
        project_level_directory = self.get_project_directory()
        onnx_model_directory = os.path.join(project_level_directory, 'onnx_model')

        if self.residual_connection:
            if self.quantization_option:
                onnx_module_path = f"{onnx_model_directory}/backup/{self.model_name}_quantized_int8_res"
            else:
                onnx_module_path = f"{onnx_model_directory}/backup/{self.model_name}_unquantized_res"
        else:
            if self.quantization_option:
                onnx_module_path = f"{onnx_model_directory}/backup/{self.model_name}_quantized_int8_seq"
            else:
                onnx_module_path = f"{onnx_model_directory}/backup/{self.model_name}_unquantized_seq"

        self.onnx_module_to_split_path = onnx_module_path

        input_for_export = tokenizer("This is a test input for exporting model", return_tensors="pt")["input_ids"]

        self.test_input = input_for_export.cpu().numpy().copy()
        self.input_for_flop = tokenizer("University of California Irvine "
                                        "is a public university located in", return_tensors="pt")[
            "input_ids"].cpu().numpy().copy()
        print(self.module_flop_map)

        if not os.path.isdir(onnx_module_path):
            os.makedirs(onnx_module_path)
        if not os.listdir(onnx_module_path):
            if self.residual_connection and self.split_size >= 2:
                if sequential_dependency_map is not None:
                    torch_to_onnx_residual(module_list=modules,
                                           input=input_for_export,
                                           export_path=onnx_module_path,
                                           sequential_dependency_map=sequential_dependency_map,
                                           residual_dependency_map=residual_dependency_map,
                                           transformer_option=self.transformer_model_option,
                                           quantization_option=self.quantization_option)
                else:
                    raise RuntimeError(f"Sequential Dependency Map cannot be None!")
            else:
                torch_to_onnx(modules, input_for_export, onnx_module_path,
                              transformer_option=self.transformer_model_option,
                              quantization_option=self.quantization_option)

        time.sleep(0.5)
        gc.collect()

    def prepare_model_split_optimized(self, module_arrangement: list):
        if os.path.exists(self.onnx_module_to_split_path):
            shutil.rmtree(self.onnx_module_to_split_path)
        # Setup seed
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        # load model to cpu
        model, tokenizer = self.load_model_and_tokenizer()
        self.tokenizer = tokenizer

        model_alloc, device_module_map = create_model_alloc(module_arrangement)
        self.device_module_arrangement = []
        for i in device_module_map.keys():
            self.device_module_arrangement.append(device_module_map[i]["fixed"] + device_module_map[i]["dynamic"])
        print(self.device_module_arrangement)
        # split model based on max number model split available
        split_size = max_split_size(model, transformer_model_option=self.transformer_model_option)
        self.split_size = split_size

        sequential_dependency_map = None
        residual_dependency_map = None
        if self.residual_connection:
            modules, sequential_dependency_map, residual_dependency_map, checked_split_size = model_sharding(model,
                                                                                                             split_size,
                                                                                                             transformer_model_option=self.transformer_model_option,
                                                                                                             residual_connection=self.residual_connection,
                                                                                                             debug_mod=False,
                                                                                                             model_allocation=model_alloc,
                                                                                                             split_option="optimized")
            self.sequential_dependency_map = sequential_dependency_map
            self.residual_dependency_map = residual_dependency_map
        else:
            modules, checked_split_size = model_sharding(model,
                                                         split_size,
                                                         transformer_model_option=self.transformer_model_option,
                                                         residual_connection=self.residual_connection,
                                                         debug_mod=False,
                                                         model_allocation=model_alloc,
                                                         split_option="optimized"
                                                         )

        self.split_size = checked_split_size
        # create input for model export
        project_level_directory = self.get_project_directory()
        onnx_model_directory = os.path.join(project_level_directory, 'onnx_model')

        if self.residual_connection:
            if self.quantization_option:
                onnx_module_path = f"{onnx_model_directory}/backup/{self.model_name}_quantized_int8_res"
            else:
                onnx_module_path = f"{onnx_model_directory}/backup/{self.model_name}_unquantized_res"
        else:
            if self.quantization_option:
                onnx_module_path = f"{onnx_model_directory}/backup/{self.model_name}_quantized_int8_seq"
            else:
                onnx_module_path = f"{onnx_model_directory}/backup/{self.model_name}_unquantized_seq"

        self.onnx_module_to_split_path = onnx_module_path

        input_for_export = tokenizer("This is a test input for exporting model", return_tensors="pt")["input_ids"]

        self.test_input = input_for_export.cpu().numpy().copy()
        self.input_for_flop = tokenizer("University of California Irvine "
                                        "is a public university located in", return_tensors="pt")[
            "input_ids"].cpu().numpy().copy()

        if not os.path.isdir(onnx_module_path):
            os.makedirs(onnx_module_path)
        if not os.listdir(onnx_module_path):
            if self.residual_connection and self.split_size >= 2:
                if sequential_dependency_map or residual_dependency_map:
                    torch_to_onnx_residual(module_list=modules,
                                           input=input_for_export,
                                           export_path=onnx_module_path,
                                           sequential_dependency_map=sequential_dependency_map,
                                           residual_dependency_map=residual_dependency_map,
                                           transformer_option=self.transformer_model_option,
                                           quantization_option=self.quantization_option)
                else:
                    raise RuntimeError(f"Sequential Dependency Map and Residual Dependency Map cannot be None!")
            else:
                torch_to_onnx(modules, input_for_export, onnx_module_path,
                              transformer_option=self.transformer_model_option,
                              quantization_option=self.quantization_option)

    def prepare_device_flops(self):
        """
            Method for preparing the information for computing device flops

            Return:
                1. bytearray_saving_path - str: serialized bytearray for device to run pseduo-inference.
                2. flop_test_module_saving_path - str: model path that's sent to device for inference.
                3. test_module_flops - int: number of model flops.
        """
        if not self.onnx_module_to_split_path:
            self.prepare_model_split()

        max_flop_module_index = self.max_flop_module_index_val[0]
        max_flops = self.max_flop_module_index_val[1]  # max_flops will be used for computing the
        model_directories = [d for d in os.listdir(self.onnx_module_to_split_path) if d.startswith('module')]
        num_models = len(model_directories)
        session_list = []
        name_list = []

        print("PREPARING MODULE FLOP INFOS.\n")
        for rank in range(num_models):
            name_path = f"{self.onnx_module_to_split_path}/module{rank}/module_name_{rank}.json"
            module_path = f"{self.onnx_module_to_split_path}/module{rank}/"

            session, name = initialize_onnx_model(module_path, name_path, rank,
                                                  quantization_option=self.quantization_option)
            session_list.append(session)
            name_list.append(name)

        input_for_inference = self.input_for_flop

        if self.residual_connection:
            buffer = InferenceBuffer(self.sequential_dependency_map, self.residual_dependency_map)
            for index, submodule in enumerate(session_list):
                if index == 0:
                    current_inputs = input_for_inference
                    output = onnx_inference_distributed([current_inputs], submodule, name_list[index])
                    buffer.update(index, forward_output=output)
                elif index == len(session_list) - 1:  # handle the last submodule edge case
                    current_inputs = buffer.get_inputs_for_submodule(index)
                    output = onnx_inference_distributed(current_inputs, submodule, name_list[index])
                else:
                    current_inputs = buffer.get_inputs_for_submodule(index)
                    output = onnx_inference_distributed(current_inputs, submodule, name_list[index])
                    buffer.update(index, output)
                if index == max_flop_module_index:
                    module = onnx_module_loading(f"{self.onnx_module_to_split_path}/module{index}/", index,
                                                 self.quantization_option)
                    input_onnx_types = [node.type.tensor_type.elem_type for node in module.graph.input]
                    # serialize_tensors expects a list of tensors; wrap if raw numpy array
                    input_tensors = current_inputs if isinstance(current_inputs, list) else [current_inputs]
                    test_tensor_bytearray = serialize_tensors(input_tensors, input_onnx_types)
                    onnx.save(module, f"{self.onnx_module_to_split_path}/flop_test_module.onnx")
                    bytearray_saving_path = os.path.join(self.onnx_module_to_split_path, "flop_byte_array.bin")
                    with open(bytearray_saving_path, 'wb') as f:
                        f.write(test_tensor_bytearray)
                    return bytearray_saving_path, f"{self.onnx_module_to_split_path}/flop_test_module.onnx", max_flops
        else:
            for rank in range(num_models):
                if rank == max_flop_module_index:
                    module = onnx_module_loading(f"{self.onnx_module_to_split_path}/module{rank}/", rank,
                                                 self.quantization_option)
                    input_onnx_types = [node.type.tensor_type.elem_type for node in module.graph.input]
                    input_tensors = input_for_inference if isinstance(input_for_inference, list) else [input_for_inference]
                    test_tensor_bytearray = serialize_tensors(input_tensors, input_onnx_types)
                    onnx.save(module, f"{self.onnx_module_to_split_path}/flop_test_module.onnx")
                    bytearray_saving_path = os.path.join(self.onnx_module_to_split_path, "flop_byte_array.bin")
                    with open(bytearray_saving_path, 'wb') as f:
                        f.write(test_tensor_bytearray)
                    return bytearray_saving_path, f"{self.onnx_module_to_split_path}/flop_test_module.onnx", max_flops

                if rank == 0:
                    input_for_inference = onnx_inference_distributed([input_for_inference], session_list[rank],
                                                                     name_list[rank])
                else:
                    input_for_inference = onnx_inference_distributed(input_for_inference, session_list[rank],
                                                                     name_list[rank])
        print("MODULE FLOP INFO PREPARATION FINISHED.\n")

    def profiling_hardware_util(self):
        # TODO: This method needs to be fixed. Currently, the memory usage for the onnx model loading is incorrect.
        #   Junchen 2023/09/18.
        module_to_memory_profile = {}
        if not self.onnx_module_to_split_path:
            self.prepare_model_split()
        model_directories = [d for d in os.listdir(self.onnx_module_to_split_path) if d.startswith('module')]
        num_models = len(model_directories)
        session_list = []
        name_list = []

        print("PROFILING MODULE MEMORY CONSUMPTION.\n")
        for rank in range(num_models):
            gc.collect()
            time.sleep(0.05)  # A short delay to let OS updates memory status

            name_path = f"{self.onnx_module_to_split_path}/module{rank}/module_name_{rank}.json"
            module_path = f"{self.onnx_module_to_split_path}/module{rank}/"

            session, name = initialize_onnx_model(module_path, name_path, rank,
                                                  quantization_option=self.quantization_option)
            model_memory_overhead = get_model_size(model_path=module_path)
            session_list.append(session)
            name_list.append(name)

            gc.collect()
            time.sleep(0.05)

            if rank in module_to_memory_profile:
                module_to_memory_profile[rank]['load'].append(model_memory_overhead)
            else:
                module_to_memory_profile[rank] = {'load': [model_memory_overhead], 'run': []}

        input_for_inference = self.test_input

        if self.residual_connection:
            buffer = InferenceBuffer(self.sequential_dependency_map, self.residual_dependency_map)
            for index, submodule in enumerate(session_list):

                gc.collect()  # Trigger garbage collection to free up any unreleased memory
                time.sleep(0.05)
                before_run = get_memory_usage()

                if index == 0:  # handle the first submodule edge case
                    current_inputs = input_for_inference
                    output = onnx_inference_distributed([current_inputs], submodule, name_list[index])
                    buffer.update(index, forward_output=output)
                elif index == len(session_list) - 1:  # handle the last submodule edge case
                    current_inputs = buffer.get_inputs_for_submodule(index)
                    output = onnx_inference_distributed(current_inputs, submodule, name_list[index])
                else:
                    current_inputs = buffer.get_inputs_for_submodule(index)
                    output = onnx_inference_distributed(current_inputs, submodule, name_list[index])
                    buffer.update(index, output)

                gc.collect()  # Trigger garbage collection after inference
                time.sleep(0.05)
                after_run = get_memory_usage()

                memory_increase = after_run - before_run
                module_to_memory_profile[index]['run'].append(memory_increase)
        else:
            for rank in range(num_models):
                gc.collect()  # Trigger garbage collection to free up any unreleased memory
                time.sleep(0.05)
                before_run = get_memory_usage()

                if rank == 0:
                    input_for_inference = onnx_inference_distributed([input_for_inference], session_list[rank],
                                                                     name_list[rank])
                else:
                    input_for_inference = onnx_inference_distributed(input_for_inference, session_list[rank],
                                                                     name_list[rank])

                gc.collect()  # Trigger garbage collection after inference
                time.sleep(0.05)
                after_run = get_memory_usage()

                memory_increase = after_run - before_run
                module_to_memory_profile[rank]['run'].append(memory_increase)

        # Calculate and print the average memory usage
        for rank in range(num_models):
            module_to_memory_profile[rank]['load'] = sum(module_to_memory_profile[rank]['load'])
            module_to_memory_profile[rank]['run'] = sum(module_to_memory_profile[rank]['run'])

        print("PROFILING MODULE MEMORY CONSUMPTION IS FINISHED.\n")
        with open(f"{self.onnx_module_to_split_path}/memory_utils.json", "w") as file:
            json.dump(module_to_memory_profile, file)

        return f"{self.onnx_module_to_split_path}/memory_utils.json", module_to_memory_profile

    def prepare_optimization_info(self):
        """
            function for collecting module memory consumption info and resources for testing device flops

            Return:
                1. mem_util
                2. output_size_map
                3. flop_byte_array_pth - str: string path for the byte array fake input to the flop test module on
                    device. It's saved as a binary file.
                4. flop_test_module_pth - str: flop test module that's going to be sent to devices for testing flop/s.
                5. test_module_flops - int: the number of flops of the flop test module.
                5. module_flop_map - dict: the flops for each module
                7. num_modules -  int: the number of submodules in total after split
        """
        _, mem_util = self.profiling_hardware_util()
        flop_byte_array_pth, flop_test_module_pth, test_module_flops = self.prepare_device_flops()
        _, output_size_map = model_output_size(self.tokenizer,
                                               self.onnx_module_to_split_path,
                                               self.split_size,
                                               self.quantization_option,
                                               self.residual_connection,
                                               self.sequential_dependency_map,
                                               self.residual_dependency_map)

        num_modules = self.split_size
        return [mem_util, output_size_map,
                flop_byte_array_pth, flop_test_module_pth,
                test_module_flops, self.module_flop_map, num_modules
                ]

    def prepare_model_to_send(self, module_arrangement: list):
        """
            Arg:
            1. module_arrangement - list: A multi-dimensional array storing how each device should receive their
            modules correspondingly.

            Output:
            1. out_dir - list: a list of sorted module path str for retreiving the module path.
        """
        self.prepare_model_split_optimized(module_arrangement=module_arrangement)

        split_model_name = self.onnx_module_to_split_path.split("/")[-1]
        project_level_directory = self.get_project_directory()
        onnx_model_directory = os.path.join(project_level_directory, 'onnx_model')
        if self.residual_connection:
            model_to_send_directory = f"{onnx_model_directory}/to_send/{split_model_name}"
        else:
            model_to_send_directory = f"{onnx_model_directory}/to_send/{split_model_name}"

        if not os.path.exists(model_to_send_directory):
            os.makedirs(model_to_send_directory)

        existing_device_dirs = [x for x in os.listdir(model_to_send_directory)
                                if re.search(r'\d+', x) and
                                os.path.isdir(os.path.join(model_to_send_directory, x))]
        if not existing_device_dirs or len(existing_device_dirs) != len(module_arrangement):
            for d in existing_device_dirs:
                shutil.rmtree(os.path.join(model_to_send_directory, d))
            device_module_assignment_optimized(self.onnx_module_to_split_path, model_to_send_directory,
                                               self.device_module_arrangement, self.quantization_option,
                                               self.sequential_dependency_map,
                                               self.residual_dependency_map,
                                               load_balancing_option=self.load_balancing_option)

        device_dir = [f"{model_to_send_directory}/{submodule}" for submodule in
                      sorted([x for x in os.listdir(model_to_send_directory) if re.search(r'\d+', x)],
                             key=lambda x: int(re.search(r'\d+', x).group(0)))]

        print(f'device_dir: {device_dir}')

        # Replace sending onnx to sending to zipped model directory to fix the onnx protobuf issue
        out_dir = []
        model_dir = []
        for device_sub_dir in device_dir:
            model_dir.append([f"{device_sub_dir}/{submodule}" for submodule in
                              sorted([x for x in os.listdir(device_sub_dir) if re.search(r'\d+', x)],
                                     key=lambda x: int(re.search(r'\d+', x).group(0)))])

        # start zipping model files
        if not self.load_balancing_option:
            print("\nZIPPING MODULES FOR SENDING TO EACH DEVICES.")
            model_zipped_dir = []
            for device_index in tqdm(range(len(model_dir))):
                temp_dir = []
                for module_dir in model_dir[device_index]:
                    source_dir = module_dir
                    tokenizer_dir = self.retreive_tokenizer_path()  # save model tokenizer to the zipping directory
                    shutil.copy(tokenizer_dir, source_dir)
                    module_zip_name = "module.zip"
                    target_dir = os.path.join(module_dir, module_zip_name)
                    temp_dir.append(source_dir)
                    print(f"zipping module file in {source_dir}.")
                    zip_directory(source_dir, target_dir)
                    print("zipping finished")
                model_zipped_dir.append(temp_dir)

            for device_module_dir in model_zipped_dir:
                temp_dir = []
                for dirs in device_module_dir:
                    submodule_dir = dirs
                    onnx_files = [f for f in os.listdir(submodule_dir) if f.endswith('.zip')]
                    if onnx_files:
                        onnx_file = onnx_files[0]
                        submodule_dir = os.path.join(submodule_dir, onnx_file)
                    temp_dir.append(submodule_dir)
                out_dir.append(temp_dir)

            print(out_dir)
            return out_dir
        else:
            # start zipping device files
            up_dir = os.path.dirname(device_dir[0])

            # copy tokenizer to device0
            tokenizer_dir = self.retreive_tokenizer_path()
            print(f"tokenizer_dir: {tokenizer_dir}")
            shutil.copy(tokenizer_dir, device_dir[0])

            for device_sub_dir in device_dir:
                device_zip_name = "device.zip"
                target_zip_dir = os.path.join(device_sub_dir, device_zip_name)

                zip_directory(device_sub_dir, target_zip_dir)
                out_dir.append(target_zip_dir)
            print(out_dir)
            return out_dir

    def retreive_tokenizer_path(self):
        cached_model_name = available_models[self.model_name][0]
        cached_model_name = "models--" + cached_model_name
        cached_model_name = cached_model_name.replace("/", "--")
        cached_model_name = f"{default_cache_path}/{cached_model_name}/snapshots/"
        tokenizer_directory = ""
        for subdir in os.listdir(cached_model_name):
            subdir_path = os.path.join(cached_model_name, subdir)
            if os.path.isdir(subdir_path):
                # Check for the presence of a file that is commonly part of a tokenizer, such as tokenizer.json
                if "tokenizer.json" in os.listdir(subdir_path):
                    tokenizer_directory = f"{subdir_path}/tokenizer.json"
                    break
                if "vocab.json" in os.listdir(subdir_path):
                    tokenizer_directory = f"{subdir_path}/tokenizer.json"
                    vocab_path = f"{subdir_path}/vocab.json"
                    # Copy vocab.json to tokenizer.json
                    shutil.copy(vocab_path, tokenizer_directory)
                    break
                if "tokenizer.model" in os.listdir(subdir_path):
                    self.tokenizer.save_pretrained(subdir_path)
                    tokenizer_directory = f"{subdir_path}/tokenizer.json"
                    break
        return tokenizer_directory


def zip_directory(source_dir, target_zip):
    # For now, we are not compressing the model file therefore, we choose zipfile.ZIP_STORE as compression option.
    with zipfile.ZipFile(target_zip, 'w', zipfile.ZIP_STORED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)

                # Check if the file being zipped is the same as the target_zip file
                if file_path == target_zip:
                    continue  # Skip zipping the target zip file itself

                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname=arcname)

    # Delete the original files after zipping
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)

            # Check if the file being deleted is not the target zip file
            if file_path != target_zip:
                os.remove(file_path)


def retrieve_sending_info(root_dir, model_name, ip_module_list, quantization_option, residual_connection):
    ip_graph = [ip_module[0] for ip_module in ip_module_list]
    if residual_connection:
        quantized_option_path = f'{model_name}_quantized_int8_res' if quantization_option else f'{model_name}_unquantized_res'
    else:
        quantized_option_path = f'{model_name}_quantized_int8_seq' if quantization_option else f'{model_name}_unquantized_seq'

    directory_path = os.path.join(root_dir, 'onnx_model', 'backup', quantized_option_path)
    _ensure_seq_dep_maps(directory_path)
    dependency_map = {"send_seq": f"{directory_path}/sender_seq_dep_map.json",
                      "send_res": f"{directory_path}/sender_res_dep_map.json",
                      "rece_seq": f"{directory_path}/receiver_seq_dep_map.json",
                      "rece_res": f"{directory_path}/receiver_res_dep_map.json"}

    return ip_graph, dependency_map


def _ensure_seq_dep_maps(directory_path):
    """Auto-generate sequential dep maps from model_input/output_names.json if missing."""
    sender_seq = os.path.join(directory_path, 'sender_seq_dep_map.json')
    sender_res = os.path.join(directory_path, 'sender_res_dep_map.json')
    receiver_seq = os.path.join(directory_path, 'receiver_seq_dep_map.json')
    receiver_res = os.path.join(directory_path, 'receiver_res_dep_map.json')

    if all(os.path.isfile(p) for p in [sender_seq, sender_res, receiver_seq, receiver_res]):
        return  # already present

    output_names_path = os.path.join(directory_path, 'model_output_names.json')
    if not os.path.isfile(output_names_path):
        raise FileNotFoundError(f"Cannot auto-generate dep maps: {output_names_path} missing")

    with open(output_names_path) as f:
        output_names = json.load(f)  # {"module_0": [...], "module_1": [...]}

    # Build sender/receiver maps: each module i sends all its outputs to module i+1
    modules = sorted(output_names.keys(), key=lambda k: int(k.split('_')[1]))
    sender_seq_data = {}
    receiver_seq_data = {}
    for idx in range(len(modules) - 1):
        src = str(idx)
        dst = str(idx + 1)
        num_outputs = len(output_names[modules[idx]])
        tensor_indices = list(range(num_outputs))
        sender_seq_data[src] = {dst: tensor_indices}
        receiver_seq_data[dst] = {src: tensor_indices}

    for path, data in [(sender_seq, sender_seq_data), (sender_res, {}),
                       (receiver_seq, receiver_seq_data), (receiver_res, {})]:
        with open(path, 'w') as f:
            json.dump(data, f)
    print(f"[retrieve_sending_info] Auto-generated dep maps in {directory_path}")


def retrieve_sending_dir(root_dir, model_name, quantization_option, residual_connection):
    if residual_connection:
        quantized_option_path = f'{model_name}_quantized_int8_res' if quantization_option else f'{model_name}_unquantized_res'
    else:
        quantized_option_path = f'{model_name}_quantized_int8_seq' if quantization_option else f'{model_name}_unquantized_seq'

    directory_path = os.path.join(root_dir, 'onnx_model', 'to_send', quantized_option_path)

    return directory_path


def retrieve_file_cfg(ip_module_list):
    file_cfg = {}
    for ip_module in ip_module_list:
        file_cfg[ip_module[0].encode("utf-8")] = ip_module[1]

    return file_cfg
