# UniChart: A Universal Vision-language Pretrained Model for Chart Comprehension and Reasoning

* Authors: [Ahmed Masry](https://ahmedmasryku.github.io/)*, Parsa Kavehzadeh*, Do Long, Shafiq Joty, Enamul Hoque (*equal contribution)
* Paper Link: [UniChart](https://arxiv.org/abs/2305.14761)
* **[NEW]** If you are looking for more advanced Chart Models, explore our latest models for chart understanding:
    * [ChartInstruct](https://github.com/vis-nlp/ChartInstruct)
        * Our advanced Chart Large Language Model based on LLaVA, supporting LLama2 (7B) and Flan-T5-XL (3B). Perfect for a wide range of chart-related tasks.
    * [ChartGemma](https://github.com/vis-nlp/ChartGemma)
        * The state-of-the-art Chart LLM built on PaliGemma (3B), optimized for visual reasoning tasks. 	
    * **Both models are user-friendly and can be run with just a few lines of code. Public web demos are available! Check out their GitHub repositories for more details.**

## UniChart Pretraining Dataset
Our pretraining dataset is divided into two primary components:
1. A zip file encompassing all the images. You can access the images through this huggingface dataset: [Images](https://huggingface.co/datasets/ahmed-masry/UniChart-pretrain-images)
2. A Huggingface dataset containing the input/output pairs utilized for model pretraining. You can find the dataset here: [Huggingface Dataset](https://huggingface.co/datasets/ahmed-masry/unichart-pretrain-data)

## UniChart Model Checkpoints
We release the checkpoints for our pretrained models as well as the finetuned checkpoints on the different downstream tasks
| Task  | Checkpoint Path |
| ------------- | ------------- |
| Pretrained  | [unichart-base-960](https://huggingface.co/ahmed-masry/unichart-base-960)  |
| ChartQA  | [unichart-chartqa-960](https://huggingface.co/ahmed-masry/unichart-chartqa-960)  |
| Chart2Text-Statista  | [unichart-chart2text-statista-960](https://huggingface.co/ahmed-masry/unichart-chart2text-statista-960)  |
| Chart2Text-Pew  | [unichart-chart2text-pew-960](https://huggingface.co/ahmed-masry/unichart-chart2text-pew-960)  |
| OpenCQA  | [unichart-opencqa-960](https://huggingface.co/ahmed-masry/unichart-opencqa-960)  |

## Web Demo
If you wish to quickly try our models, you can access our public web demoes hosted on the Hugging Face Spaces platform with a friendly interface!

| Tasks  | Web Demo |
| ------------- | ------------- |
| Base Model (Best for Chart Summarization and Data Table Generation)  | [UniChart-Base](https://huggingface.co/spaces/ahmed-masry/UniChart-Base) |
| Chart Question Answering  | [UniChart-ChartQA](https://huggingface.co/spaces/ahmed-masry/UniChart-ChartQA) |

The input prompt for Chart summarization is **<summarize_chart>** and Data Table Generation is **<extract_data_table>**

## Requirements

```
transformers==4.28.1
pytorch-lightning==1.8.5
datasets
sentencepiece
```
Please make sure to use the **exact same version** of the **Transformers** library. We have noticed that there might be a drop in performance when using different versions of the library! 
## Inference
You can easily use our models for inference with the huggingface library! 
You just need to do the following:
1. Change _model_name_ to your prefered checkpoint.
2. Chage the _imag_path_ to your chart example image path on your system
3. Write the _input_prompt_ based on your prefered task as shown in the table below.

| Task  | Input Prompt |
| ------------- | ------------- |
| Chart Question Answering  | \<chartqa\> question <s_answer>  |
| Open Chart Question Answering  | \<opencqa\> question <s_answer>  |
| Chart Summarization  | <summarize_chart> <s_answer>  |
| Data Table Extraction  | <extract_data_table> <s_answer>  |

```
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch, os, re

torch.hub.download_url_to_file('https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_1229.png', 'chart_example_1.png')

model_name = "ahmed-masry/unichart-chartqa-960"
image_path = "/content/chart_example_1.png"
input_prompt = "<chartqa> What is the lowest value in blue bar? <s_answer>"

model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = DonutProcessor.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

image = Image.open(image_path).convert("RGB")
decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt").input_ids
pixel_values = processor(image, return_tensors="pt").pixel_values

outputs = model.generate(
    pixel_values.to(device),
    decoder_input_ids=decoder_input_ids.to(device),
    max_length=model.decoder.config.max_position_embeddings,
    early_stopping=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    num_beams=4,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)
sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = sequence.split("<s_answer>")[1].strip()
print(sequence)

```

## Finetuning 
In order to finetune the model on the ChartQA dataset, you can edit and run the following command:
```
python finetune_chartqa.py --data-path "ahmed-masry/chartqa_without_images" --train-images '/content/ChartQA/ChartQA Dataset/train/png/' \
    --valid-images '/content/ChartQA/ChartQA Dataset/val/png' --max-steps 40000 --batch-size 8 --valid-batch-size 1 --num-workers 12 --lr 5e-5 \
    --check-val-every-n-epoch 1 --warmup-steps 100 --checkpoint-steps 7000 --checkpoint-path "ahmed-masry/unichart-base-960"
```

# Contact
If you have any questions about this work, please contact **[Ahmed Masry](https://ahmedmasryku.github.io/)** using the following email addresses: **amasry17@ku.edu.tr** or **ahmed.elmasry24653@gmail.com**.

# Reference
Please cite our paper if you use our models or dataset in your research. 

```
@misc{masry2023unichart,
      title={UniChart: A Universal Vision-language Pretrained Model for Chart Comprehension and Reasoning}, 
      author={Ahmed Masry and Parsa Kavehzadeh and Xuan Long Do and Enamul Hoque and Shafiq Joty},
      year={2023},
      eprint={2305.14761},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
