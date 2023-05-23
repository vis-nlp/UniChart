# UniChart: A Universal Vision-language Pretrained Model for Chart Comprehension and Reasoning

* Authors: Ahmed Masry*, Parsa Kavehzadeh*, Do Long, Shafiq Joty, Enamul Hoque
* Paper Link: [UniChart]()

## Pretraining Dataset
Coming soon.


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

## Inference
You can easily use our models for inference with the huggingface library! 
You just need to do the following:
1. Change _model_name_ to your prefered checkpoint.
2. Chage the _imag_path_ to your chart example image path on your system
3. Write the _input_prompt_ based on your prefered task as shown in the table below.

| Task  | Input Prompt |
| ------------- | ------------- |
| Chart Question Answering  | <chartqa> <s_answer>  |
| Open Chart Question Answering  | <opencqa> <s_answer>  |
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

## Training 
Coming soon.

# Contact
If you have any questions about this work, please contact **Ahmed Masry** using the following email addresses: **amasry17@ku.edu.tr** or **ahmed.elmasry24653@gmail.com**.

# Reference
Please cite our paper if you use our models or dataset in your research. 

```
Coming soon
```
