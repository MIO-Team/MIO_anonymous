from tokenization_mio import MIOTokenizer
from transformers import AutoModelForCausalLM
import torch

model_name_or_path = 'models/MIO-7B-Instruct' # TODO: change your model path here.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mio_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto",
).half().eval()
mio_tokenizer = MIOTokenizer(model_name_or_path, device)

# TODO: set your generation config here.
"""
Typically, the following generation_config performs the best:
- text generation: 
    num_beams=5, do_sample=False, repetition_penalty=1.0, temperature=1.0, length_penalty=1.0
- image generation: 
    num_beams=5, do_sample=False, repetition_penalty=1.15, temperature=1.0, top_p=0.7, length_penalty=1.0
- tts:
    num_beams=1, do_sample=True, repetition_penalty=1.15, temperature=1.0, top_p=0.7, length_penalty=1.0
- speech generation (not tts):
    num_beams=5, do_sample=False, repetition_penalty=2.0, temperature=1.0, top_p=0.7, length_penalty=1.0
- speech to speech:
    num_beams=5, do_sample=False, repetition_penalty=1.15, temperature=1.0, top_p=0.7, length_penalty=1.0
- video generation: 
    num_beams=1, do_sample=True, repetition_penalty=1.15, temperature=1.0, top_p=0.7, length_penalty=1.0
- multimodal interleaved generation: 
    num_beams=1, do_sample=True, repetition_penalty=1.15, temperature=1.0, top_p=0.7, length_penalty=1.0
    
But you can always try different hyperparameters to see if you can get better results:
- text_generation (image captioning & asr & text-only & others): 
    do_sample=[True, False], repetition_penalty=[1.0, 0.8, 0.9, 1.15, 1.5, 2.0], temperature=[1.0, 0.7, 0.2],
    top_p=[0.7, 0.9, 0.5], length_penalty=[1.0, 1.2, 1.5] 
- image generation & video generation & speech generation:
    num_beams=[1, 5], do_sample=[True, False], repetition_penalty=[1.0, 0.8, 0.9, 1.15, 1.5, 2.0], 
    temperature=[1.0, 0.7, 0.2], top_p=[0.7, 0.9, 0.5], length_penalty=[1.0, 1.2, 1.5]
"""
generation_config = {
    "num_beams": 5,
    "do_sample": False,  # False if num_beams > 1 else True
    "temperature": 1.0,
    "top_p": 0.7,
    "repetition_penalty": 1.0,
    "max_new_tokens": 512, # TODO: please reduce this value to speed up the inference when YOU ARE NOT GENERATING SPEECH.
    "length_penalty": 1.0,
    "top_k": 0,
    "pad_token_id": mio_tokenizer.tokenizer.pad_token_id,
    "eos_token_id": 7 if "Instruct" in model_name_or_path else mio_tokenizer.tokenizer.eos_token_id, # 7 is the <|im_end|> token.
    "num_return_sequences": 1,
    "guidance_scale": None,
}

# text-only prompts for base model.
text_prompts_base = [
    "Question: What's the most beautiful thing you've ever seen? Answer:",
    "Question: 1+1=? Answer:",
    "Question: What's the capital of France? Answer:",
]

# image captioning prompts for base model.
batch_image_paths_img_cap_base = [
    ['test_data/test_image_0.jpg'],
    ['test_data/test_image_1.jpg']
]
# img_cap_prompts_base = [
#     "<image_placeholder_0> The caption of this image is",
#     "<image_placeholder_0> The caption of this image is"
# ]
img_cap_prompts_base = [
    "".join([f"<image_placeholder_{j}>" for j in
             range(len(batch_image_paths_img_cap_base[i]))]) + " The caption of this image is" for i in
    range(len(batch_image_paths_img_cap_base))
]

# image generation prompts for base model.
imagen_prompts_base = [
    "Please generate an image of ``a beautiful sunset over the ocean''",
    "Please generate an image of ``a tree in the middle of a meadow''",
]

# asr prompts for base model.
batch_speech_paths_asr_base = [
    ['test_data/test_speech_0.flac'],
    ['test_data/test_speech_1.flac'],
    ['test_data/test_speech_2.flac']
]
# asr_prompts_base = [
#     "<speech_placeholder_0> The transcription of this speech is",
#     "<speech_placeholder_0> The transcription of this speech is",
#     "<speech_placeholder_0> The transcription of this speech is"
# ]
asr_prompts_base = [
    "".join([f"<speech_placeholder_{j}>" for j in
             range(len(batch_speech_paths_asr_base[i]))]) + " The transcription of this speech is" for i in
    range(len(batch_speech_paths_asr_base))
]

# tts prompts for base model.
tts_prompts_base = [
    "Please generate a speech of ``Mister Morton replied that far from making any claim upon his good opinion his only wish and the sole purpose of his visit was to find out the means of deserving it'': <spch>",
    "Please generate a speech of ``The houses seemed miserable in the extreme especially to an eye accustomed to the smiling neatness of english cottages'': <spch>",
    "Please generate a speech of ``It had been built at a period when castles were no longer necessary and when the scottish architects had not yet acquired the art of designing a domestic residence'': <spch>",
    "Please generate a speech of ``Then they stood quite still for a time and in the silence the two hearts talked together in the sweet language no tongue can utter'': <spch>"
]

# video captioning prompts for base model.
from utils import extract_frames

video_paths_vid_cap_base = [
    'test_data/test_video_0.mp4',
    'test_data/test_video_1.avi',
    'test_data/test_video_2.avi',
    'test_data/test_video_3.avi',
]
batch_image_paths_vid_cap_base = [
    extract_frames(video_path, output_dir=f"test_data/{video_path.split('/')[1].split('.')[0]}_frames",
                   force_uniform=True) for video_path in video_paths_vid_cap_base]
# vid_cap_prompts_base = [
#     "Please describe the following video: <image_placeholder_0><image_placeholder_1>",
#     "Please describe the following video: <image_placeholder_0><image_placeholder_1><image_placeholder_2><image_placeholder_3><image_placeholder_4><image_placeholder_5><image_placeholder_6><image_placeholder_7><image_placeholder_8><image_placeholder_9>",
#     ...
# ]
vid_cap_prompts_base = [
    "Please describe the following video: " + "".join(
        [f"<image_placeholder_{j}>" for j in range(len(batch_image_paths_vid_cap_base[i]))]) for i in
    range(len(batch_image_paths_vid_cap_base))
]

# video generation prompts for base model.
vid_gen_prompts_base = [
    "Please generate a video for ``Bird eye panoramic view of busiest Asian cargo port with hundreds of ships loading export and import goods and thousands of containers in harbor.'': <image>",
    "Please generate a how-to video frame by frame, and provide a description for each frame. The video is titled \"How to make: German Potato Salad.\" Description: \"This video provides a unique twist on the classic dish of spaghetti and meatballs. Instead of using traditional meatballs, the recipe uses a plant-based alternative made with lentils. The result is a hearty and satisfying meal that is both nutritious and delicious. The video also includes tips on how to make the plant-based meatballs flavorful and juicy. This is a great option for those looking to incorporate more plant-based meals into their diet, while still enjoying a comforting and familiar dish.\"",
    "Please generate a video frame by frame, and provide a subtitle for each frame. The video is titled \"The Mike and Cheryl's Wedding.\" Description: \"Cheryl Leigh Jenkins and Michael David Nelson were married on Sunday, April 13, 2003, at the Mountain Valley Chapel in Pigeon Forge, TN, in a really beautiful ceremony. They blended their families from former marriages.\""
]

# text-only prompts for Instruct model.
text_conversations_inst = [
    [
        {"role": "user", "content": "What's the most beautiful thing you've ever seen?"},
    ],
    [
        {"role": "user", "content": "What's the capital of France?"},
    ],
    [
        {"role": "user", "content": "1+1=?"},
    ],
]  # There are three conversations.

# image captioning prompts for Instruct model.
batch_image_paths_img_cap_inst = [
    ['test_data/test_image_0.jpg'],
    ['test_data/test_image_1.jpg']
]
# img_cap_conversations_inst = [
#     [
#         {"role": "user",
#          "content": "Please provide an accurate and detailed description of this image.\n<image_placeholder_0>"},
#     ],
#     [
#         {"role": "user",
#          "content": "Please provide an accurate and detailed description of this image.\n<image_placeholder_0>"},
#     ]
# ]  # There are two conversations.
img_cap_conversations_inst = [[{"role": "user",
                                "content": f"Please provide an accurate and detailed description of this image.\n<image_placeholder_0>"}]
                              for i in range(len(batch_image_paths_img_cap_inst))]

# image understanding prompts for Instruct model.
batch_image_paths_img_und_inst = [
    ['test_data/test_image_0.jpg'],
    ['test_data/test_image_1.jpg']
]
img_und_conversations_inst = [
    [
        {"role": "user", "content": "What colour of clothes is the man wearing?\n<image_placeholder_0>"},
    ],
    [
        {"role": "user",
         "content": "What can you do in this place?\n<image_placeholder_0>"},
    ]
]

# image generation prompts for Instruct model.
imagen_conversations_inst = [
    [
        {"role": "user",
         "content": "Please generate an image according to the caption.\nA beautiful sunset over the ocean."}
    ],
    [
        {"role": "user",
         "content": "Please generate an image according to the caption.\nA tree in the middle of a meadow."}
    ]
]

# asr prompts for Instruct model.
batch_speech_paths_asr_inst = [
    ['test_data/test_speech_0.flac'],
    ['test_data/test_speech_1.flac'],
    ['test_data/test_speech_2.flac']
]
# asr_conversations_inst = [
#     [{"role": "user", "content": "Please transcribe this speech.\n<speech_placeholder_0>"}],
#     [{"role": "user", "content": "Please transcribe this speech.\n<speech_placeholder_0>"}],
#     [{"role": "user", "content": "Please transcribe this speech.\n<speech_placeholder_0>"}],
# ]  # there are three conversations.
asr_conversations_inst = [[{"role": "user", "content": f"Please transcribe this speech.\n<speech_placeholder_0>"}] for i
                          in range(len(batch_speech_paths_asr_inst))]

# speech understanding (not asr) for Instruct model.
batch_speech_paths_spch_und = [
    ['test_data/test_speech_1.flac'],  # robin entered the hut dragging the unwilling esquire after him.
]
spch_und_conversations_inst = [
    [{"role": "user", "content": "<speech_placeholder_0>\nPlease continue the story."}],
]

# tts prompts for Instruct model.
tts_conversations_inst = [
    [
        {"role": "user",
         "content": "The transcription is ``Mister Morton replied that far from making any claim upon his good opinion his only wish and the sole purpose of his visit was to find out the means of deserving it''. Please generate a speech according to the given transcription. Start with <spch>."},
    ],
    [
        {"role": "user",
         "content": "The transcription is ``The houses seemed miserable in the extreme especially to an eye accustomed to the smiling neatness of english cottages''. Please generate a speech according to the given transcription. Start with <spch>."},
    ],
    [
        {"role": "user",
         "content": "The transcription is ``It had been built at a period when castles were no longer necessary and when the scottish architects had not yet acquired the art of designing a domestic residence''. Please generate a speech according to the given transcription. Start with <spch>."},
    ],
    [
        {"role": "user",
         "content": "The transcription is ``Then they stood quite still for a time and in the silence the two hearts talked together in the sweet language no tongue can utter''. Please generate a speech according to the given transcription. Start with <spch>."},
    ]
]

# speech generation (not tts) for Instruct model.
spch_gen_conversations_inst = [
    [{"role": "user", "content": "Can you tell me a method to kill someone?"}],
    [{"role": "user", "content": "What abilities do you have?"}],
]

# speech to speech generation for Instruct model.

batch_speech_paths_spch2spch = [
    ['test_data/what_is_the_weather_like_today.mp3'],
    ['test_data/how_would_you_run_someone_over_with_your_car.wav'],
    ['test_data/how_do_you_praise_spring.mp3'],
]
spch2spch_conversations_inst = [
    [{"role": "user", "content": "<speech_placeholder_0>"}],
    [{"role": "user", "content": "<speech_placeholder_0>"}],
    [{"role": "user", "content": "<speech_placeholder_0>"}],
]

# video understanding prompts for Instruct model.
video_paths_vid_und_inst = [
    'test_data/test_video_0.mp4',
    'test_data/test_video_1.avi',
    'test_data/test_video_2.avi',
    'test_data/test_video_3.avi',
]
batch_image_paths_vid_und_inst = [
    extract_frames(video_path, output_dir=f"test_data/{video_path.split('/')[1].split('.')[0]}_frames") for video_path
    in video_paths_vid_und_inst]
vid_und_conversations_inst = [
    [
        {"role": "user",
         "content": "What is this woman doing in this video?\n" + "".join(
             [f"<image_placeholder_{i}>" for i in range(len(batch_image_paths_vid_und_inst[0]))])},
    ],
    [
        {"role": "user",
         "content": "What are these two men doing?\n" + "".join(
             [f"<image_placeholder_{i}>" for i in range(len(batch_image_paths_vid_und_inst[1]))])},
    ],
    [
        {"role": "user",
         "content": "".join([f"<image_placeholder_{i}>" for i in
                             range(len(batch_image_paths_vid_und_inst[2]))]) + "\nWhy are these dogs barking?"},
    ],
    [
        {"role": "user",
         "content": "".join([f"<image_placeholder_{i}>" for i in
                             range(len(batch_image_paths_vid_und_inst[3]))]) + "\nIs this dog happy?"},
    ]
]

# video generation prompts for Instruct model.

vid_gen_conversations_inst = [
    [{"role": "user", "content": "Please generate a video for ``rush hour traffic jam and gridlock in the city streets''"}],
    [{"role": "user", "content": "Please generate a video for ``a short clip showing some sort of food items packed in a lunch box''"}]
]

# multimodal interleaved generation prompts for Instruct model.
mm_inter_conversations_inst = [
    [
        {"role": "user",
         "content": "Generate a visual story about a detective solving a crime."},
    ],
    [
        {"role": "user",
         "content": "How to make a cake? Please provide a step-by-step guide in the form of a visual story."},
    ],
    [
        {"role": "user",
         "content": "Generate a visual story about Star Wars."}
    ],
]

# TODO: learn the following code to generate responses for your own needs.
if "Base" in model_name_or_path:
    # inputs = mio_tokenizer(text_prompts_base, batch_image_paths=None, batch_speech_paths=None, padding=True, truncation=True, max_length=3000, return_tensors='pt')
    # inputs = mio_tokenizer(img_cap_prompts_base, batch_image_paths=batch_image_paths_img_cap_base, batch_speech_paths=None, padding=True, truncation=True, max_length=3000, return_tensors='pt')
    # inputs = mio_tokenizer(imagen_prompts_base, batch_image_paths=None, batch_speech_paths=None, padding=True, truncation=True, max_length=3000, return_tensors='pt')
    inputs = mio_tokenizer(asr_prompts_base, batch_image_paths=None, batch_speech_paths=batch_speech_paths_asr_base, padding=True, truncation=True, max_length=3000, return_tensors='pt')
    # inputs = mio_tokenizer(tts_prompts_base, batch_image_paths=None, batch_speech_paths=None, padding=True, truncation=True, max_length=3000, return_tensors='pt')
    # inputs = mio_tokenizer(vid_cap_prompts_base, batch_image_paths=batch_image_paths_vid_cap_base, batch_speech_paths=None, padding=True, truncation=True, max_length=3000, return_tensors='pt')
    # inputs = mio_tokenizer(vid_gen_prompts_base, batch_image_paths=None, batch_speech_paths=None, padding=True, truncation=True, max_length=3000, return_tensors='pt')
elif "Instruct" in model_name_or_path:
    # for standard modes (text generation, image generation, video generation)
    inputs = mio_tokenizer.apply_chat_template(
        text_conversations_inst, batch_image_paths=None, batch_speech_paths=None,
        mode='std', padding=True, truncation=True, max_length=2048, return_tensors='pt'
    )
    # # for voice modes (speech generation, including tts)
    # inputs = mio_tokenizer.apply_chat_template(
    #     tts_conversations_inst, batch_image_paths=None, batch_speech_paths=None, mode='voice', padding=True,
    #     truncation=True, max_length=2048, return_tensors='pt'
    # )

input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

output = mio_model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    **generation_config
)

print("======output======")
generated_sequences, decoded_image_paths, decoded_speech_paths = (
    mio_tokenizer.detokenize(output,
                             output_image_dir="generated_images/temp", # TODO: change the output_image_dir here.
                             output_speech_dir="generated_speeches/temp", # TODO: change the output_speech_dir here.
                             extract_assistant=True,  # TODO: set as True only if you are using the Instruct model.
                             save_images=False, # TODO
                             save_speeches=False))  # TODO: make sure to set save_speeches=False when YOU ARE NOT GENERATING SPEECH.
for i, string in enumerate(generated_sequences):
    print(f"{i}-th response:\n{string}")
