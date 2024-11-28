 
<h1 align="center">
MIO: A Foundation Model on Multimodal Tokens
</h1>

## Environment Setup

```bash
cd MIO
conda create -n mio python=3.10
conda activate mio
pip install -r requirements.txt
```

## File Structure

- `tokenization_mio.py`: (1) image/speech preprocessing, quantization, and decoding; (2) multimodal tokenization and detokenization; (3) applying the chat template.
- `utils.py`: extracting the frames from the video (both keyframe extraction and uniform frame extraction).
- `infer.py`: inference script for MIO with the examples.
- `/image_tokenizer`
- `/speech_tokenizer`

## Run Inference

Please read the TODOs and examples in the `infer.py` script to understand how to run the inference for each modality.

```bash
python infer.py
```

1. Set the most appropriate generation config (it's recommended to conduct a hyperparameter search).
2. Pay attention to the input data structures, formats, instructions, and the prompt templates.
3. Tokenize the input data and don't forget to apply the chat template in the suitable mode (`voice` v.s. `std`).
4. Generate the responses.
5. Detokenize the responses and save the results (`detokenized_{modality}_{sample_id}_{image/speech_index}.{suffix}`).