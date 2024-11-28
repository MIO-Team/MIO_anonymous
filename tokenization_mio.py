import sys
import os
from transformers import LlamaTokenizer
from image_tokenizer.models.seed_llama_tokenizer import SeedLlamaTokenizer
from image_tokenizer.models.transforms import get_transform
from speech_tokenizer import SpeechTokenizer
from omegaconf import OmegaConf
from PIL import Image
import re
import torchaudio
from scipy.io.wavfile import write
import torch
import torchvision.transforms as T

MIO_CHAT_TEMPLATE = {
    "format_system": "<|im_start|>system\n{system}<|im_end|>",
    "format_user": "<|im_start|>user\n{query}<|im_end|>",
    "format_assistant": "<|im_start|>assistant\n{response}<|im_end|>",
    "format_separator": '\n',
    "stop_words": ["<|im_end|>"],
    "assistant_starter": "<|im_start|>assistant\n",
    "user_starter": "<|im_start|>user\n",
}

MIO_VOICE_MODE_SYSTEM = "You are MIO, an AI assistant capable of understanding images, text, videos, and speech, and generating speech. Please respond to the user with speech only, starting with <spch> and ending with </spch>."
MIO_STD_MODE_SYSTEM = "You are MIO, an AI assistant capable of understanding and generating images, text, videos, and speech, selecting the appropriate modality according to the context."

def check_speech_completeness(speech_str):
    tokens = re.findall(r'<spch\d+>', speech_str)
    if not tokens:
        raise ValueError(f"Cannot found <spch\d+> tokens")

    token_info = []
    for token in tokens:
        match = re.match(r'<spch(\d+)>', token)
        if not match:
            raise ValueError(f"Cannot extract token number from the speech token {token}.")
        token_number = int(match.group(1))

        if 0 <= token_number <= 1023:
            part = 'content'
        elif 1024 <= token_number <= 2047:
            part = 'timbre0'
        elif 2048 <= token_number <= 3071:
            part = 'timbre1'
        elif 3072 <= token_number <= 4095:
            part = 'timbre2'
        else:
            raise ValueError(f"Token number <spch{token_number}> out of valid ranges of speech tokens.")

        token_info.append((token, part))

    parts = []
    current_part = token_info[0][1]
    part_tokens = [token_info[0][0]]

    for i in range(1, len(token_info)):
        token, part = token_info[i]
        if part == current_part:
            part_tokens.append(token)
        else:
            parts.append((current_part, part_tokens))
            current_part = part
            part_tokens = [token]
    parts.append((current_part, part_tokens))

    expected_order = ['content', 'timbre0', 'timbre1', 'timbre2']
    if len(parts) != 4:
        raise ValueError(f"Must have exactly four parts in the speech tokens, only found {[each[0] for each in parts]}")
    for i, (part_name, tokens_in_part) in enumerate(parts):
        if part_name != expected_order[i]:
            raise ValueError(f"Part {i} is expected to be {expected_order[i]}, but found {part_name}. Parts are not in the correct order: {[each[0] for each in parts]}")

    token_counts = [len(tokens_in_part) for _, tokens_in_part in parts]
    if len(set(token_counts)) != 1:
        raise ValueError(f"Each part must have the same number of tokens, but found {token_counts}.")

    return True



class MIOTokenizer():
    def __init__(self,
                 model_name_or_path,
                 device=None):
        self.device = device

        self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        self.image_transform = self.load_image_transform()
        self.image_tokenizer = self.load_image_tokenizer(model_name_or_path)
        self.speech_tokenizer = self.load_speech_tokenizer(model_name_or_path)

        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "right"

    def load_image_transform(self):
        transform_cfg = OmegaConf.load("image_tokenizer/configs/clip_transform.yaml")
        transform = get_transform(type=transform_cfg['type'], image_size=transform_cfg['image_size'],
                                  keep_ratio=transform_cfg['keep_ratio'])
        return transform

    def load_image_tokenizer(self, model_name_or_path):
        config = OmegaConf.load("image_tokenizer/configs/seed_llama_tokenizer.yaml")
        pretrained_model_name_or_path = os.path.join(model_name_or_path, config['pretrained_model_name_or_path'])
        encoder_path = os.path.join(model_name_or_path, config['encoder_path'])
        diffusion_path = os.path.join(model_name_or_path, config['diffusion_path'])

        tokenizer = SeedLlamaTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            fp16=config['fp16'], encoder_url=encoder_path,
            diffusion_path=diffusion_path, device=self.device,
            load_diffusion=True)

        return tokenizer

    def load_speech_tokenizer(self, model_name_or_path):
        config_path = os.path.join(model_name_or_path, "SpeechTokenizer", 'config.json')
        ckpt_path = os.path.join(model_name_or_path, "SpeechTokenizer", 'SpeechTokenizer.pt')
        tokenizer = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path).to(self.device)
        return tokenizer

    def tokenize_images(self, images, add_begin_end=True):
        if isinstance(images[0], str):
            images = [Image.open(image_path).convert('RGB') for image_path in images]
        images = [self.image_transform(image).to(self.device) for image in images]

        batch_image_ids = []
        for image in images:
            image_ids = self.image_tokenizer.encode_image(image_torch=image)  # in tensor
            image_ids = image_ids[0].tolist()
            batch_image_ids.append(image_ids)

        batch_image_str = []
        for image_ids in batch_image_ids:
            image_tokens = [f"<img{i}>" for i in image_ids]
            if add_begin_end:
                image_str = "<image>" + "".join(image_tokens) + "</image>"
            else:
                image_str = "".join(image_tokens)
            batch_image_str.append(image_str)

        return batch_image_str

    def decode_images(self, image_strs, output_image_paths, additional_info=None):

        assert len(image_strs) == len(output_image_paths)

        if isinstance(image_strs, str):
            image_strs = [image_strs]

        for i, image_str in enumerate(image_strs):
            # possible input 1: "<image><img804><img680><img356></image>"
            # possible input 2: "<image> <img804> <img680> <img356> </image>"
            # possible input 3: "<img804> <img680> <img356>"
            # possible input 4: "<img804><img680><img356>"
            image_ids = re.findall(r'\d+', image_str)
            image_ids = [int(id) for id in image_ids]
            image_ids = torch.tensor(image_ids, device=self.device).unsqueeze(0)
            rec_image = self.image_tokenizer.decode_image(image_ids)[0]
            rec_image = T.ToPILImage()(rec_image.cpu().detach()) if not isinstance(rec_image,
                                                                                   Image.Image) else rec_image
            rec_image.save(output_image_paths[i])
            if additional_info:
                print(f"{additional_info} The {i}-th image is saved in {output_image_paths[i]}. ")
            else:
                print(f"The {i}-th image is saved in {output_image_paths[i]}.")

    def tokenize_speeches(self, speech_paths, add_timbre=False, add_begin_end=True):
        batch_speech_str = []
        vocab_size = self.speech_tokenizer.config.get('codebook_size')  # 1024
        for speech_path in speech_paths:
            wav, sr = torchaudio.load(speech_path)
            wav = wav.unsqueeze(0).to(self.device)
            if sr != self.speech_tokenizer.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.speech_tokenizer.sample_rate)
            codes = self.speech_tokenizer.encode(wav)
            RVQ_1 = codes[:1, :, :]  # Contain content info, can be considered as semantic tokens
            RVQ_supplement = codes[1:4, :, :]  # Contain timbre info, complete info lost by the first quantizer
            # both in (n_q, batch, timesteps)
            if not add_timbre:
                # from (n_q, batch, timesteps) to [0, 1, 2, 3] and then to "<spch> <spch0> <spch1> <spch2> <spch3> </spch>"
                content_tokens = [f"<spch{int(tokenid)}>" for tokenid in RVQ_1[0][0].tolist()]
                if add_begin_end:
                    speech_str = "<spch>" + "".join(content_tokens) + "</spch>"
                else:
                    speech_str = " ".join(content_tokens)

                batch_speech_str.append(speech_str)
            else:
                content_tokens = [f"<spch{int(tokenid)}>" for tokenid in RVQ_1[0][0].tolist()]
                timbre0_tokens = [f"<spch{int(tokenid) + vocab_size * 1}>" for tokenid in RVQ_supplement[0][0].tolist()]
                timbre1_tokens = [f"<spch{int(tokenid) + vocab_size * 2}>" for tokenid in RVQ_supplement[1][0].tolist()]
                timbre2_tokens = [f"<spch{int(tokenid) + vocab_size * 3}>" for tokenid in RVQ_supplement[2][0].tolist()]
                if add_begin_end:
                    speech_str = "<spch>" + "".join(content_tokens) + "" + "".join(timbre0_tokens) + "" + "".join(
                        timbre1_tokens) + "" + "".join(timbre2_tokens) + "</spch>"
                else:
                    speech_str = "".join(content_tokens) + "" + "".join(timbre0_tokens) + "" + "".join(
                        timbre1_tokens) + "" + "".join(timbre2_tokens)
                batch_speech_str.append(speech_str)
        return batch_speech_str

    def decode_speeches(self, speech_strs, output_speech_paths, additional_info=None):
        """
        Decode speech tokens back into speech files and save them.

        Args:
            speech_strs (list of str): List of speech token strings to decode.
            output_speech_paths (list of str): List of file paths to save the decoded speech files.
        """
        assert len(speech_strs) == len(output_speech_paths)

        if isinstance(speech_strs, str):
            speech_strs = [speech_strs]

        for i, speech_str in enumerate(speech_strs):
            if '> <' not in speech_str:
                speech_str = speech_str.replace('><', '> <')

            tokens = speech_str.split()
            ids = []


            for token in tokens:
                if token not in ['<spch>', '</spch>']:
                    ids.append(int(token[5:-1]))

            vocab_size = self.speech_tokenizer.config.get('codebook_size')  # 1024

            if not check_speech_completeness(speech_str):
                raise ValueError(f"Cannot decode speech: The speech string {speech_str} is incomplete. Make sure that the speech string contains four consecutive parts: "
                                 f"content (<spch0>-<spch1023>), timbre0 (<spch1024>-<spch2047>), timbre1 (<spch2048>-<spch3071>), and timbre2 (<spch3072>-<spch4095>). "
                                 f"And each part is of the same length.")

            current_idx = 0
            for j, id in enumerate(ids):
                assert id < vocab_size + current_idx * vocab_size
                assert id >= 0 + current_idx * vocab_size
                next_id = ids[j + 1] if j + 1 < len(ids) else vocab_size + current_idx * vocab_size
                if next_id >= vocab_size + current_idx * vocab_size:
                    current_idx += 1

            ids = torch.tensor(ids).view(4, -1)
            ids[1] = ids[1] - vocab_size
            ids[2] = ids[2] - vocab_size * 2
            ids[3] = ids[3] - vocab_size * 3
            to_decode = ids.unsqueeze(1)
            decoded_speech = self.speech_tokenizer.decode(to_decode.to(self.device))

            write(output_speech_paths[i], self.speech_tokenizer.sample_rate, decoded_speech.cpu().detach().numpy())

            if additional_info:
                print(f"{additional_info} The {i}-th speech is saved in {output_speech_paths[i]}. ")
            else:
                print(f"The {i}-th speech is saved in {output_speech_paths[i]}.")

    def __call__(self, texts, batch_image_paths=None, batch_speech_paths=None, add_bos=True, add_timbre=False, *args, **kwargs):
        """
        Tokenize the input texts, images, and speeches.
        :param texts:
            a batch of texts, ["This is an image <image_placeholder_0>, and this is a speech <speech_placeholder_0>, ...", ...]
        :param batch_image_paths:
            a batch of image_paths, in formats supported by PIL.Image.open.
            [["", ""], ["", ""]]
        :param batch_speech_paths:
            a batch of speech_paths, better as .wav files.
            [["", ""], ["", ""]]
        """
        if batch_image_paths is not None:
            batch_image_strs = []
            for image_paths in batch_image_paths:
                images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
                image_strs = self.tokenize_images(images)
                batch_image_strs.append(image_strs)
        else:
            batch_image_strs = [[] for _ in range(len(texts))]

        if batch_speech_paths is not None:
            batch_speech_strs = []
            for speech_paths in batch_speech_paths:
                speech_strs = self.tokenize_speeches(speech_paths, add_timbre=add_timbre)
                batch_speech_strs.append(speech_strs)
        else:
            batch_speech_strs = [[] for _ in range(len(texts))]

        sequences = []

        for i in range(len(texts)):
            sequence = texts[i]
            for j, image_str in enumerate(batch_image_strs[i]):
                sequence = sequence.replace(f"<image_placeholder_{j}>", image_str)
            for j, speech_str in enumerate(batch_speech_strs[i]):
                sequence = sequence.replace(f"<speech_placeholder_{j}>", speech_str)

            if not sequence.startswith(self.tokenizer.bos_token) and add_bos:
                sequence = self.tokenizer.bos_token + sequence

            sequences.append(sequence)

        tokenized = self.tokenizer(sequences, *args, **kwargs)

        return tokenized

    def detokenize(self, tokenized_input_ids, output_image_dir="decoded_images", output_speech_dir="decoded_speeches", save_images=True, save_speeches=True, extract_assistant=False):
        """
        Detokenize the tokenized output back into the original input format (texts, decoded images, and decoded speech).

        Args:
            tokenized_input_ids (tensor): input_ids of the tokenized inputs.
            output_image_dir (str): Directory to save decoded images.
            output_speech_dir (str): Directory to save decoded speech.

        Returns:
            tuple: A tuple containing:
                - texts (list of str): The reconstructed original texts.
                - decoded_image_paths (list of list of str): Paths to the decoded images.
                - decoded_speech_paths (list of list of str): Paths to the decoded speech.
        """
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_speech_dir, exist_ok=True)

        decoded_sequences = self.tokenizer.batch_decode(tokenized_input_ids, skip_special_tokens=False)
        if extract_assistant:
            new_decoded_sequences = []
            for seq in decoded_sequences:
                final_response = seq.split(MIO_CHAT_TEMPLATE["assistant_starter"])[-1].replace("<|im_end|>", "").replace("<unk>", "").strip()
                new_decoded_sequences.append(final_response)
            decoded_sequences = new_decoded_sequences

        texts = []
        decoded_image_paths = []
        decoded_speech_paths = []

        for idx, seq in enumerate(decoded_sequences):
            text = seq
            image_paths = []
            speech_paths = []

            image_placeholders = re.findall(r"(<image>\s*(?:<img\d+>(?:\s*<img\d+>)*)\s*</image>)", text)
            if image_placeholders and save_images:
                output_image_paths = [os.path.join(output_image_dir, f"detokenized_image_{idx}_{i}.jpg")
                                      for i in range(len(image_placeholders))]
                self.decode_images(image_placeholders, output_image_paths, additional_info=f"{idx}-th sample in batch.")
                image_paths.extend(output_image_paths)
                for i, placeholder in enumerate(image_placeholders):
                    text = text.replace(placeholder, f"<image_placeholder_{i}>", 1)

            speech_placeholders = re.findall(r"(<spch>\s*(?:<spch\d+>(?:\s*<spch\d+>)*)\s*</spch>)", text)
            if speech_placeholders and save_speeches:
                output_speech_paths = [os.path.join(output_speech_dir, f"detokenized_speech_{idx}_{i}.wav")
                                       for i in range(len(speech_placeholders))]

                speech_placeholders = [speech_placeholder for speech_placeholder in speech_placeholders if check_speech_completeness(speech_placeholder)]
                self.decode_speeches(speech_placeholders, output_speech_paths, additional_info=f"{idx}-th sample in batch.")
                speech_paths.extend(output_speech_paths)
                for i, placeholder in enumerate(speech_placeholders):
                    text = text.replace(placeholder, f"<speech_placeholder_{i}>", 1)

            texts.append(text)
            decoded_image_paths.append(image_paths)
            decoded_speech_paths.append(speech_paths)

        return texts, decoded_image_paths, decoded_speech_paths

    def apply_chat_template(self, conversations, batch_image_paths=None, batch_speech_paths=None, chat_template=None,
                            add_generation_prompt=True, tokenize=True, mode="std", add_bos=True, add_timbre=False, *args, **kwargs):
        # conversations: [[{"role": "user", "content": "xxx"}, {"role": "assistant", "content": "xxx"}], ...]
        if conversations[0] is dict:
            conversations = [conversations]

        conv_strs = []
        for conversation in conversations:
            conv_str = ""
            if chat_template is None:
                chat_template = MIO_CHAT_TEMPLATE

            if mode == "std":
                system_str = MIO_STD_MODE_SYSTEM
            elif mode == "voice":
                system_str = MIO_VOICE_MODE_SYSTEM
            elif mode == "custom":
                assert conversation[0]["role"] == "system", "In custom mode, the first turn must be the system."
                system_str = conversation[0]["content"]
            else:
                raise ValueError(
                    f"Unknown mode: {mode}, must choose from ['std', 'voice'].\nThe std system is: {MIO_STD_MODE_SYSTEM}\nThe voice system is: {MIO_VOICE_MODE_SYSTEM}")

            conv_str += chat_template["format_system"].format(system=system_str) + chat_template["format_separator"]

            for turn in conversation:
                if turn["role"] == "user":
                    conv_str += chat_template["format_user"].format(query=turn["content"]) + chat_template[
                        "format_separator"]
                elif turn["role"] == "assistant":
                    conv_str += chat_template["format_assistant"].format(response=turn["content"]) + chat_template[
                        "format_separator"]
                elif turn["role"] == "system":
                    pass
                else:
                    raise ValueError(f"Unknown role: {turn['role']}")

            if add_generation_prompt:
                # add assistant_starter
                conv_str += chat_template["assistant_starter"]
            else:
                conv_str = conv_str[:-len(chat_template["format_separator"])]

            conv_strs.append(conv_str)

        if tokenize:
            return self(conv_strs, batch_image_paths, batch_speech_paths, add_bos=add_bos, add_timbre=add_timbre, *args, **kwargs)
        else:
            if add_bos:
                return [self.tokenizer.bos_token + each for each in conv_strs]
            else:
                return conv_strs


if __name__ == '__main__':
    mio_tokenizer = MIOTokenizer('models/MIO-7B-Base', device='cuda:0')

    # # test tokenize images
    # test_images = [
    #     "test_data/test_image_0.jpg",
    #     "test_data/test_image_1.jpg"
    # ]
    # print(mio_tokenizer.tokenize_images(test_images))

    # # test decode images
    # test_image_strs = [
    #     ' <image>  <img6589>  <img680>  <img5943>  <img7927>  <img680>  <img256>  <img2189>  <img4815>  <img6665>  <img2157>  <img2157>  <img4966>  <img4966>  <img2189>  <img4815>  <img2050>  <img2157>  <img7487>  <img2157>  <img4815>  <img4966>  <img7927>  <img2189>  <img2157>  <img2050>  <img7927>  <img339>  <img3845>  <img2189>  <img7927>  <img4815>  <img3845>  </image>',
    #     '<image> <img5828> <img3449> <img3896> <img7857> <img2315> <img853> <img4781> <img6941> <img5252> <img680> <img680> <img2977> <img4781> <img5826> <img3352> <img3051> <img2157> <img4388> <img2157> <img3051> <img4781> <img6284> <img1031> <img2157> <img6084> <img3051> <img2952> <img2191> <img3896> <img2297> <img7857> <img665> </image>']
    # test_output_image_paths = [
    #     "test_data/recon_test_image_0.jpg",
    #     "test_data/recon_test_image_1.jpg"
    # ]
    # mio_tokenizer.decode_images(test_image_strs, test_output_image_paths)

    # test tokenize speeches
    test_speeches = [
        "test_data/test_speech_0.flac",
        "test_data/test_speech_1.flac",
        "test_data/test_speech_2.flac",
    ]
    speech_tokens = mio_tokenizer.tokenize_speeches(test_speeches, add_timbre=True) # add_timbre = False for speech input, True for speech output
    print(speech_tokens)
    print(len(speech_tokens)) # 2
    print(len(speech_tokens[0].split())) # 3922

    # test decode speeches
    # speech_tokens = [
    #
    # ]
    speech_output_paths = [
        "test_data/recon_test_speech_0.wav",
        "test_data/recon_test_speech_1.wav",
        "test_data/recon_test_speech_2.wav",
    ]
    mio_tokenizer.decode_speeches(speech_tokens, speech_output_paths)

    # # test tokenize
    # texts = ["The boy is eating <image_placeholder_0>. The girl is eating <image_placeholder_1>. The mom is saying <speech_placeholder_0>.",
    #          "The dog is eating <image_placeholder_1>. The cat is eating <image_placeholder_0>. The grandma is saying <speech_placeholder_0>."]
    # batch_image_paths = [
    #     ["test_data/test_image_0.jpg", "test_data/test_image_1.jpg"],
    #     ["test_data/test_image_1.jpg", "test_data/test_image_0.jpg"]
    # ]
    # batch_speech_paths = [
    #     ["test_data/test_speech_0.flac"],
    #     ["test_data/test_speech_1.flac"]
    # ]
    #
    # tokenized = mio_tokenizer(texts, batch_image_paths=batch_image_paths, batch_speech_paths=batch_speech_paths, add_timbre=True) # add_timbre = False for speech input, True for speech output
    # # print(tokenized)
    #
    # # test detokenize
    # texts, decoded_image_paths, decoded_speech_paths = mio_tokenizer.detokenize(tokenized['input_ids'])
    # print(texts)
    # print(decoded_image_paths)
    # print(decoded_speech_paths)

    # # test apply_chat_template
    # conversations = [
    #     [{"role": "user", "content": "The boy is eating <image_placeholder_0>. What is the girl eating?"}],
    #     [{"role": "user", "content": "What am I eating?"},
    #      {"role": "assistant", "content": "You are eating <image_placeholder_0>."},
    #      {"role": "user", "content": "What is the girl eating?"}]
    # ]
    # batch_image_paths = [
    #     ["test_data/test_image_0.jpg"],
    #     ["test_data/test_image_1.jpg"]
    # ]
    # # std_tokenized = mio_tokenizer.apply_chat_template(conversations, batch_image_paths=batch_image_paths, tokenize=False, mode="std")
    # # print("std_tokenized:", std_tokenized)
    # voice_tokenized = mio_tokenizer.apply_chat_template(conversations, batch_image_paths=batch_image_paths, tokenize=True, add_timbre=True, mode="voice")
    # voice_detokenized = mio_tokenizer.detokenize(voice_tokenized)
    # print("voice_tokenized:", voice_detokenized)