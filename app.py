import json

import PIL
import gradio as gr
import numpy as np
from gradio import processing_utils

from packaging import version
from PIL import Image

from caption_anything.model import CaptionAnything
from caption_anything.utils.image_editing_utils import create_bubble_frame
from caption_anything.utils.utils import mask_painter, seg_model_map, prepare_segmenter, is_platform_win
from caption_anything.utils.parser import parse_augment
from caption_anything.captioner import build_captioner
from caption_anything.text_refiner import build_text_refiner
from caption_anything.segmenter import build_segmenter
from caption_anything.utils.chatbot import ConversationBot, build_chatbot_tools, get_new_image_name
from segment_anything import sam_model_registry

args = parse_augment()

if args.segmenter_checkpoint is None:
    _, segmenter_checkpoint = prepare_segmenter(args.segmenter)
else:
    segmenter_checkpoint = args.segmenter_checkpoint

shared_captioner = build_captioner(args.captioner, args.device, args)
shared_sam_model = sam_model_registry[seg_model_map[args.segmenter]](checkpoint=segmenter_checkpoint).to(args.device)


class ImageSketcher(gr.Image):
    """
    Fix the bug of gradio.Image that cannot upload with tool == 'sketch'.
    """

    is_template = True  # Magic to make this work with gradio.Block, don't remove unless you know what you're doing.

    def __init__(self, **kwargs):
        super().__init__(tool="sketch", **kwargs)

    def preprocess(self, x):
        if self.tool == 'sketch' and self.source in ["upload", "webcam"]:
            assert isinstance(x, dict)
            if x['mask'] is None:
                decode_image = processing_utils.decode_base64_to_image(x['image'])
                width, height = decode_image.size
                mask = np.zeros((height, width, 4), dtype=np.uint8)
                mask[..., -1] = 255
                mask = self.postprocess(mask)

                x['mask'] = mask

        return super().preprocess(x)


def build_caption_anything_with_models(args, api_key="", captioner=None, sam_model=None, text_refiner=None,
                                       session_id=None):
    segmenter = build_segmenter(args.segmenter, args.device, args, model=sam_model)
    captioner = captioner
    if session_id is not None:
        print('Init caption anything for session {}'.format(session_id))
    return CaptionAnything(args, api_key, captioner=captioner, segmenter=segmenter, text_refiner=text_refiner)


def init_openai_api_key(length,sentiment,factuality,language,api_key=""):
    text_refiner = None
    if api_key and len(api_key) > 20:
        try:
            controls = {'length': length,
                        'sentiment': sentiment,
                        'factuality': factuality,
                        'language': language}

            text_refiner = build_text_refiner(args.text_refiner, args.device, args, api_key)
            text_refiner.llm('hi',controls)  # test
            print("✅ OpenAI API 初始化成功")
        except Exception as e:
            print("OpenAI init error:", e)
            text_refiner = None
    openai_available = text_refiner is not None
    return gr.update(visible=openai_available), gr.update(visible=openai_available), gr.update(
        visible=openai_available), gr.update(visible=True), gr.update(visible=True), gr.update(
        visible=True), text_refiner


def get_click_prompt(chat_input, click_state, click_mode):
    inputs = json.loads(chat_input)
    if click_mode == 'Continuous':
        points = click_state[0]
        labels = click_state[1]
        for input in inputs:
            points.append(input[:2])
            labels.append(input[2])
    elif click_mode == 'Single':
        points = []
        labels = []
        for input in inputs:
            points.append(input[:2])
            labels.append(input[2])
        click_state[0] = points
        click_state[1] = labels
    else:
        raise NotImplementedError

    prompt = {
        "prompt_type": ["click"],
        "input_point": click_state[0],
        "input_label": click_state[1],
        "multimask_output": "True",
    }
    return prompt


def update_click_state(click_state, caption, click_mode):
    if click_mode == 'Continuous':
        click_state[2].append(caption)
    elif click_mode == 'Single':
        click_state[2] = [caption]
    else:
        raise NotImplementedError

# 积分对话；图像对话功能，点击位置，每个位置的描述
# def chat_with_points(chat_input, click_state, chat_state, state, text_refiner, img_caption,length,sentiment,factuality,language):
#     if text_refiner is None:
#         response = "Text refiner is not initilzed, please input openai api key."
#         state = state + [(chat_input, response)]
#         return state, state, chat_state
#     # 解析点击状态中的坐标点、标签、图像描述文本
#     points, labels, captions = click_state
#     # point_chat_prompt = "I want you act as a chat bot in terms of image. I will give you some points (w, h) in the image and tell you what happed on the point in natural language. Note that (0, 0) refers to the top-left corner of the image, w refers to the width and h refers the height. You should chat with me based on the fact in the image instead of imagination. Now I tell you the points with their visual description:\n{points_with_caps}\nNow begin chatting!"
    
#     # point_chat_prompt = "请你扮演图像场景下的聊天机器人。我会给出图像中的若干坐标点（宽，高），
#     # 并用自然语言描述该点位对应的画面内容。请注意：坐标（0, 0）代表图像左上角，
#     # w 为宽度数值，h 为高度数值。你需要严格依据图像实际内容和我对话，切勿凭空想象。.
#     # 接下来，我将为你提供带视觉描述的坐标信息：\n{points_with_caps}\n现在开始对话！"
#     # 对话后缀模板
#     suffix = '\nHuman: {chat_input}\nAI: '
#     # 问答历史模板
#     qa_template = '\nHuman: {q}\nAI: {a}'
#     controls = {'length': length,
#                 'sentiment': sentiment,
#                 'factuality': factuality,
#                 'language': language}

#     # # "The image is of width {width} and height {height}." 
#     # 该图片的宽度为{width}，高度为{height}。
#     # 我是一款专为图像对话训练的人工智能。我擅长根据你提供的图像信息，精准解读任意画面中的内容。
#     # 图像整体描述为：{img_caption}。同时，请详细列出画面中的各类物体，包含其所处位置与视觉特征。
#     # 以下为画面中各类场景的位置及细节描述：{points_with_caps}。
#     # 请使用文字而非数字来描述所有位置信息。.现在，开始对话吧！图像点位对话提示
#     point_chat_prompt = "I am an AI trained to chat with you about an image. I am greate at what is going on in any image based on the image information your provide. " \
#     # "The overall image description is \"{img_caption}\". You will also provide me objects in the image in details, i.e., their location and visual descriptions. " \
#     # "Here are the locations and descriptions of events that happen in the image: {points_with_caps} \nYou are required to use language instead of number to describe these positions. Now, let's chat!"
#     # point_chat_prompt = """
#     # You are an AI assistant that MUST answer based ONLY on the given image information.

#     # Image description:
#     # {img_caption}

#     # Object details:
#     # {points_with_caps}

#     # Rules:
#     # - Do NOT repeat previous answers
#     # - Provide new details each time
#     # - Be more specific and descriptive

#     # Now answer the question.
#     # """
        
#     # # 初始化视觉上下文 / 有效点位 / 点位描述列表
#     prev_visual_context = ""
#     pos_points = []
#     pos_captions = []
#     # 遍历筛选有效点位信息
#     for i in range(len(points)):
#         if labels[i] == 1:
#             pos_points.append(f"(X:{points[i][0]}, Y:{points[i][1]})")
#             pos_captions.append(captions[i])
#     # 拼接点位对应的视觉事件上下文
#     # for i in range(len(pos_points)):
#     #     prev_visual_context += \
#     #         f'\nThere is an event described as "{pos_captions[i]}" locating at {pos_points[i]}'
#     prev_visual_context = prev_visual_context + '\n' + 'There is an event described as  \"{}\" locating at {}'.format(
#         pos_captions[-1], ', '.join(pos_points))
#     # 上下文长度阈值
#     context_length_thres = 500
#     # 历史对话内容
#     prev_history = ""
#     # 拼接历史对话内容，限制文本长度
#     for i in range(len(chat_state)):
#         q, a = chat_state[i]
#         if len(prev_history) < context_length_thres:
#             prev_history = prev_history + qa_template.format(**{"q": q, "a": a})
#         else:
#             break
#     # 拼接完成对话提示词：图像描述+点位信息+历史对话+当前用户输入
#     chat_prompt = point_chat_prompt.format(
#         **{"img_caption": img_caption, "points_with_caps": prev_visual_context}) + prev_history + suffix.format(
#         **{"chat_input": chat_input})
#     # 打印最终的拼接的对话提示词
#     print('\nchat_prompt: ', chat_prompt)
#     # 调用大模型生成回复
#     # response = text_refiner.inference(query=chat_prompt,controls=controls,)
#     # response = response['caption']
#     response = text_refiner.llm(chat_prompt)
#     # 更全局会话状态与聊天状态
#     state = state + [(chat_input, response)]
#     chat_state = chat_state + [(chat_input, response)]
#     # 返回更新后的状态数据
#     return state, state, chat_state

def chat_with_points(chat_input, click_state, chat_state, state, text_refiner, img_caption,length,sentiment,factuality,language):
    if text_refiner is None:
        response = "Text refiner is not initilzed, please input openai api key."
        state = state + [(chat_input, response)]
        return state, state, chat_state

    points, labels, captions = click_state
    # point_chat_prompt = "I want you act as a chat bot in terms of image. I will give you some points (w, h) in the image and tell you what happed on the point in natural language. Note that (0, 0) refers to the top-left corner of the image, w refers to the width and h refers the height. You should chat with me based on the fact in the image instead of imagination. Now I tell you the points with their visual description:\n{points_with_caps}\nNow begin chatting!"
    suffix = '\nHuman: {chat_input}\nAI: '
    qa_template = '\nHuman: {q}\nAI: {a}'
    # # "The image is of width {width} and height {height}." 
    point_chat_prompt = "I am an AI trained to chat with you about an image. I am greate at what is going on in any image based on the image information your provide. The overall image description is \"{img_caption}\". You will also provide me objects in the image in details, i.e., their location and visual descriptions. Here are the locations and descriptions of events that happen in the image: {points_with_caps} \nYou are required to use language instead of number to describe these positions. Now, let's chat!"
    prev_visual_context = ""
    pos_points = []
    pos_captions = []
    controls = {'length': length,
            'sentiment': sentiment,
            'factuality': factuality,
            'language': language}
    for i in range(len(points)):
        if labels[i] == 1:
            pos_points.append(f"(X:{points[i][0]}, Y:{points[i][1]})")
            pos_captions.append(captions[i])
    prev_visual_context = prev_visual_context + '\n' + 'There is an event described as  \"{}\" locating at {}'.format(
        pos_captions[-1], ', '.join(pos_points))

    context_length_thres = 500
    prev_history = ""
    for i in range(len(chat_state)):
        q, a = chat_state[i]
        if len(prev_history) < context_length_thres:
            prev_history = prev_history + qa_template.format(**{"q": q, "a": a})
        else:
            break
    chat_prompt = point_chat_prompt.format(
        **{"img_caption": img_caption, "points_with_caps": prev_visual_context}) + prev_history + suffix.format(
        **{"chat_input": chat_input})
    print('\nchat_prompt: ', chat_prompt)
    response = text_refiner.llm(chat_prompt,controls)
    state = state + [(chat_input, response)]
    chat_state = chat_state + [(chat_input, response)]
    return state, state, chat_state

def upload_callback(image_input, state,text_refiner):
    if isinstance(image_input, dict):  # if upload from sketcher_input, input contains image and mask
        image_input, mask = image_input['image'], image_input['mask']

    chat_state = []
    click_state = [[], [], []]
    res = 1024
    width, height = image_input.size
    ratio = min(1.0 * res / max(width, height), 1.0)
    if ratio < 1.0:
        image_input = image_input.resize((int(width * ratio), int(height * ratio)))
        print('Scaling input image to {}'.format(image_input.size))
    # state = [] + [(None, 'Image size: ' + str(image_input.size))]
    state = []
    model = build_caption_anything_with_models(
        args,
        api_key="",
        captioner=shared_captioner,
        sam_model=shared_sam_model,
        session_id=iface.app_id,
        text_refiner=text_refiner,
    )
    model.segmenter.set_image(image_input)
    image_embedding = model.image_embedding
    original_size = model.original_size
    input_size = model.input_size
    img_caption = model.captioner.inference(image_input)['caption']
    # chat_state.append(img_caption)

    return state, state, chat_state, image_input, click_state, image_input, image_input, image_embedding, \
        original_size, input_size, img_caption,model


def inference_click(image_input, point_prompt, click_mode, enable_wiki, language, sentiment, factuality,
                    length, image_embedding, state, click_state, original_size, input_size, text_refiner,
                    evt: gr.SelectData,chat_state):
    click_index = evt.index

    if point_prompt == 'Positive':
        coordinate = "[[{}, {}, 1]]".format(str(click_index[0]), str(click_index[1]))
    else:
        coordinate = "[[{}, {}, 0]]".format(str(click_index[0]), str(click_index[1]))
    
    prompt = get_click_prompt(coordinate, click_state, click_mode)
    input_points = prompt['input_point']
    input_labels = prompt['input_label']
    # 配置文本生成控制参数
    controls = {'length': length,
                'sentiment': sentiment,
                'factuality': factuality,
                'language': language}
    
    model = build_caption_anything_with_models(
        args,
        api_key="",
        captioner=shared_captioner,
        sam_model=shared_sam_model,
        text_refiner=text_refiner,
        session_id=iface.app_id
    )

    model.setup(image_embedding, original_size, input_size, is_image_set=True)

    enable_wiki = True if enable_wiki in ['True', 'TRUE', 'true', True, 'Yes', 'YES', 'yes'] else False
    disable_gpt = False if text_refiner else True
    out = model.inference(image_input, prompt, controls, disable_gpt=disable_gpt, enable_wiki=enable_wiki)[0]

    # state = state + [("Image point: {}, Input label: {}".format(prompt["input_point"], prompt["input_label"]), None)]
    state = state + [(None, "raw_caption: {}".format(out['generated_captions']['raw_caption']))]
    wiki = out['generated_captions'].get('wiki', "")
    update_click_state(click_state, out['generated_captions']['raw_caption'], click_mode)
    text = out['generated_captions']['raw_caption']
    # chat_state.append(text)
    input_mask = np.array(out['mask'].convert('P'))
    image_input = mask_painter(np.array(image_input), input_mask)
    origin_image_input = image_input
    image_input = create_bubble_frame(image_input, text, (click_index[0], click_index[1]), input_mask,
                                      input_points=input_points, input_labels=input_labels)
    yield state, state, click_state, image_input, wiki,chat_state
    if disable_gpt and model.text_refiner:
        refined_caption = model.text_refiner.inference(query=text, controls=controls, context=out['context_captions'],
                                                       enable_wiki=enable_wiki)
        # new_cap = 'Original: ' + text + '. Refined: ' + refined_caption['caption']
        new_cap = refined_caption['caption']
        wiki = refined_caption['wiki']
        state = state + [(None, f"caption: {new_cap}")]
        refined_image_input = create_bubble_frame(origin_image_input, new_cap, (click_index[0], click_index[1]),
                                                  input_mask,
                                                  input_points=input_points, input_labels=input_labels)
        yield state, state, click_state, refined_image_input, wiki


def get_sketch_prompt(mask: PIL.Image.Image, multi_mask=True):
    """
    Get the prompt for the sketcher.
    The mask is clustered into different groups, whose bounding boxes are returned with prompt
    """

    mask = np.array(np.asarray(mask)[..., 0])
    mask[mask > 0] = 1  # Refine the mask, let all nonzero values be 1

    if not multi_mask:
        y, x = np.where(mask == 1)
        x1, y1 = np.min(x), np.min(y)
        x2, y2 = np.max(x), np.max(y)

        prompt = {
            'prompt_type': ['box'],
            'input_boxes': [
                [x1, y1, x2, y2]
            ]
        }

        return prompt

    traversed = np.zeros_like(mask)
    groups = np.zeros_like(mask)
    max_group_id = 1

    # Iterate over all pixels
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if traversed[x, y] == 1:
                continue

            if mask[x, y] == 0:
                traversed[x, y] = 1
            else:
                # If pixel is part of mask
                groups[x, y] = max_group_id
                stack = [(x, y)]
                while stack:
                    i, j = stack.pop()
                    if traversed[i, j] == 1:
                        continue
                    traversed[i, j] = 1
                    if mask[i, j] == 1:
                        groups[i, j] = max_group_id
                        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                            ni, nj = i + di, j + dj
                            traversed[i, j] = 1
                            if 0 <= nj < mask.shape[1] and mask.shape[0] > ni >= 0 == traversed[ni, nj]:
                                stack.append((i + di, j + dj))
                max_group_id += 1

    # get the bounding box of each group
    boxes = []
    for group in range(1, max_group_id):
        y, x = np.where(groups == group)
        x1, y1 = np.min(x), np.min(y)
        x2, y2 = np.max(x), np.max(y)
        boxes.append([x1, y1, x2, y2])

    prompt = {
        'prompt_type': ['box'],
        'input_boxes': boxes
    }

    return prompt


def inference_traject(sketcher_image, enable_wiki, language, sentiment, factuality, length, image_embedding, state,
                      original_size, input_size, text_refiner,chat_state):
    image_input, mask = sketcher_image['image'], sketcher_image['mask']

    prompt = get_sketch_prompt(mask, multi_mask=True)
    boxes = prompt['input_boxes']

    controls = {'length': length,
                'sentiment': sentiment,
                'factuality': factuality,
                'language': language}

    model = build_caption_anything_with_models(
        args,
        api_key="",
        captioner=shared_captioner,
        sam_model=shared_sam_model,
        text_refiner=text_refiner,
        session_id=iface.app_id
    )

    model.setup(image_embedding, original_size, input_size, is_image_set=True)

    enable_wiki = True if enable_wiki in ['True', 'TRUE', 'true', True, 'Yes', 'YES', 'yes'] else False
    disable_gpt = False if text_refiner else True
    out = model.combined_inference(image_input, prompt, controls, disable_gpt=disable_gpt, enable_wiki=enable_wiki,
                                   enable_morphologyex=True)

    # Update components and states
    # state.append((f'Box: {boxes}', None))
    state.append((None, f'raw_caption: {out["generated_captions"]["raw_caption"]}'))
    wiki = out['generated_captions'].get('wiki', "")
    text = out['generated_captions']['raw_caption']
    # chat_state.append(text)
    input_mask = np.array(out['mask'].convert('P'))
    image_input = mask_painter(np.array(image_input), input_mask)

    origin_image_input = image_input

    fake_click_index = (int((boxes[0][0] + boxes[0][2]) / 2), int((boxes[0][1] + boxes[0][3]) / 2))
    image_input = create_bubble_frame(image_input, text, fake_click_index, input_mask)

    yield state, state, image_input, wiki,chat_state

    if disable_gpt and model.text_refiner:
        refined_caption = model.text_refiner.inference(query=text, controls=controls, context=out['context_captions'],
                                                       enable_wiki=enable_wiki)

        new_cap = refined_caption['caption']
        wiki = refined_caption['wiki']
        state = state + [(None, f"caption: {new_cap}")]
        refined_image_input = create_bubble_frame(origin_image_input, new_cap, fake_click_index, input_mask)

        yield state, state, refined_image_input, wiki


def get_style():
    current_version = version.parse(gr.__version__)
    if current_version <= version.parse('3.24.1'):
        style = '''
        #image_sketcher{min-height:500px}
        #image_sketcher [data-testid="image"], #image_sketcher [data-testid="image"] > div{min-height: 500px}
        #image_upload{min-height:500px}
        #image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 500px}
        '''
    elif current_version <= version.parse('3.27'):
        style = '''
        #image_sketcher{min-height:500px}
        #image_upload{min-height:500px}
        '''
    else:
        style = None

    return style


def create_ui():
    title = """<p><h1 align="center">Caption-Anything</h1></p>
    """
    description = """<p>Gradio演示，适用于任何字幕，字幕生成，支持多种语言风格。使用方法只需上传你的图片后点击其中一个示例加载即可，可以使用大模型进行对话</p>"""

    examples = [
        ["test_images/img35.webp"],
        ["test_images/img2.jpg"],
        ["test_images/img5.jpg"],
        ["test_images/img12.jpg"],
        ["test_images/img14.jpg"],
        ["test_images/qingming3.jpeg"],
        ["test_images/img1.jpg"],
    ]

    with gr.Blocks(
            css=get_style()
    ) as iface:
        state = gr.State([])
        click_state = gr.State([[], [], []])
        chat_state = gr.State([])
        origin_image = gr.State(None)
        image_embedding = gr.State(None)
        text_refiner = gr.State(None)
        original_size = gr.State(None)
        input_size = gr.State(None)
        img_caption = gr.State(None)
        model_state = gr.State(None)

        gr.Markdown(title)
        gr.Markdown(description)
        # 行 / 列
        with gr.Row():
            with gr.Column(scale=1.0):
                # 非GPT模块
                with gr.Column(visible=False) as modules_not_need_gpt:
                    with gr.Tab("Click"):
                        image_input = gr.Image(type="pil", interactive=True, elem_id="image_upload").style(height=500)
                        example_image = gr.Image(type="pil", interactive=False, visible=False)
                        with gr.Row(scale=1.0):
                            with gr.Row(scale=0.4):
                                point_prompt = gr.Radio(
                                    choices=["Positive", "Negative"],
                                    value="Positive",
                                    label="Point Prompt",
                                    interactive=True)
                                # 点击提示
                                click_mode = gr.Radio(
                                    choices=["Continuous", "Single"],
                                    value="Continuous",
                                    label="Clicking Mode",
                                    interactive=True)
                            with gr.Row(scale=0.4):
                                clear_button_click = gr.Button(value="Clear Clicks", interactive=True)
                                clear_button_image = gr.Button(value="Clear Image", interactive=True)
                    with gr.Tab("Trajectory (Beta)"):
                        # 草图工具输入
                        sketcher_input = ImageSketcher(type="pil", interactive=True, brush_radius=20,
                                                       elem_id="image_sketcher").style(height=500)
                        with gr.Row():
                            clear_button_traj = gr.Button(value="Clear Trajectory", interactive=True)
                            clear_button_traj_img = gr.Button(value="Clear Image", interactive=True)
                            submit_button_sketcher = gr.Button(value="Submit", interactive=True)
                # GPT
                with gr.Column(visible=False) as modules_need_gpt:
                    # 控制组件
                    with gr.Row(scale=1.0):
                        language = gr.Dropdown(
                            ['English', 'Chinese', 'French', "Spanish", "Arabic", "Portuguese", "Cantonese"],
                            value="Chinese", label="Language", interactive=True)
                        sentiment = gr.Radio(
                            choices=["Positive", "Natural", "Negative"],
                            value="Natural",
                            label="Sentiment",
                            interactive=True,
                        )
                    with gr.Row(scale=1.0):
                        factuality = gr.Radio(
                            choices=["Factual", "Imagination"],
                            value="Factual",
                            label="Factuality",
                            interactive=True,
                        )
                        length = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=36,
                            step=1,
                            interactive=True,
                            label="Generated Caption Length",
                        )
                        # enable_wiki = gr.Radio(
                        #     choices=["Yes", "No"],
                        #     value="No",
                        #     label="Enable Wiki",
                        #     interactive=True)
                        enable_wiki = gr.Radio(visible=False)
                # 示例图片区
                with gr.Column(visible=True) as modules_not_need_gpt3:
                    gr.Examples(
                        examples=examples,
                        inputs=[example_image],
                    )
            # 右侧输出区
            with gr.Column(scale=0.5):
                # API输入
                openai_api_key = gr.Textbox(
                    placeholder="Input QWEN API key",
                    show_label=False,
                    label="QWEN API Key",
                    lines=1,
                    type="password")
                # 是否启用
                with gr.Row(scale=0.5):
                    # 按钮组件
                    enable_chatGPT_button = gr.Button(value="Run with QWEN", interactive=True, variant='primary')
                    disable_chatGPT_button = gr.Button(value="Run without QWEN (Faster)", interactive=True,
                                                       variant='primary')
                # Wiki结果框
                with gr.Column(visible=False) as modules_need_gpt2:
                    # Wiki 输出
                    # wiki_output = gr.Textbox(lines=5, label="Wiki", max_lines=5)
                    wiki_output = gr.Textbox()
                # 聊天区
                with gr.Column(visible=False) as modules_not_need_gpt2:
                    chatbot = gr.Chatbot(label="Chat about Selected Object", ).style(height=550, scale=0.5)
                    # 输入框
                    with gr.Column(visible=False) as modules_need_gpt3:
                        chat_input = gr.Textbox(show_label=False, placeholder="Enter text and press Enter").style(
                            container=False)
                        # 清除/提交按钮
                        with gr.Row():
                            clear_button_text = gr.Button(value="Clear Text", interactive=True)
                            submit_button_text = gr.Button(value="Submit", interactive=True, variant="primary")
        # 初始化ai / 启用 / 禁用
        openai_api_key.submit(init_openai_api_key, inputs=[length,sentiment,factuality,language,openai_api_key],
                              outputs=[modules_need_gpt, modules_need_gpt2, modules_need_gpt3, modules_not_need_gpt,
                                       modules_not_need_gpt2, modules_not_need_gpt3, text_refiner])
        enable_chatGPT_button.click(init_openai_api_key, inputs=[length,sentiment,factuality,language,openai_api_key],
                                    outputs=[modules_need_gpt, modules_need_gpt2, modules_need_gpt3,
                                             modules_not_need_gpt,
                                             modules_not_need_gpt2, modules_not_need_gpt3, text_refiner])
        disable_chatGPT_button.click(init_openai_api_key,
                                     outputs=[modules_need_gpt, modules_need_gpt2, modules_need_gpt3,
                                              modules_not_need_gpt,
                                              modules_not_need_gpt2, modules_not_need_gpt3, text_refiner])
        # 清空图片 / 清空对话 / 清空聊天状态 / 清空点击点 / 清空wiki / 清空原图
        enable_chatGPT_button.click(
            lambda: (None, [], [], [[], [], []], "", "", ""),
            [],
            [image_input, chatbot, state, click_state, wiki_output, origin_image],
            queue=False,
            show_progress=False
        )

        disable_chatGPT_button.click(
            lambda: (None, [], [], [[], [], []], "", "", ""),
            [],
            [image_input, chatbot, state, click_state, wiki_output, origin_image],
            queue=False,
            show_progress=False
        )
        # 清除点击点
        clear_button_click.click(
            lambda x: ([[], [], []], x, ""),
            [origin_image],
            [click_state, image_input, wiki_output],
            queue=False,
            show_progress=False
        )
        # 清空整张图片
        clear_button_image.click(
            lambda: (None, [], [], [], [[], [], []], "", "", ""),
            [],
            [image_input, chatbot, state, chat_state, click_state, wiki_output, origin_image, img_caption],
            queue=False,
            show_progress=False
        )
        # 清空聊天
        clear_button_text.click(
            lambda: ([], [], [[], [], []], []),
            [],
            [chatbot, state, click_state, chat_state],
            queue=False,
            show_progress=False
        )
        # 图像组件清空
        image_input.clear(
            lambda: (None, [], [], [], [[], [], []], "", "", ""),
            [],
            [image_input, chatbot, state, chat_state, click_state, wiki_output, origin_image, img_caption],
            queue=False,
            show_progress=False
        )
        # 只清轨迹
        clear_button_traj.click(
            lambda x: (x, ""),
            [origin_image],
            [sketcher_input, wiki_output],
            queue=False,
            show_progress=False
        )
        # 清全部
        clear_button_traj_img.click(
            lambda: (None, [], [], [], "", "", ""),
            [],
            [sketcher_input, chatbot, state, chat_state, wiki_output, origin_image, img_caption],
            queue=False,
            show_progress=False
        )
        chat_input.submit(lambda: "", None, chat_input)

        # submit_button_text.click(
        #     lambda x: ("", x),
        #     inputs=[chat_input],
        #     outputs=[chat_input, state],
        #     queue=False
        # ).then(
        #     chat_with_points,
        #     [chat_input, click_state, chat_state, state, text_refiner, img_caption],
        #     [chatbot, state, chat_state]
        # )

        # # 上传图像
        image_input.upload(upload_callback, [image_input, state, text_refiner],
                           [chatbot, state, chat_state, origin_image, click_state, image_input, sketcher_input,
                            image_embedding, original_size, input_size, img_caption, model_state])
        # 涂鸦上传
        sketcher_input.upload(upload_callback, [sketcher_input, state, text_refiner],
                              [chatbot, state, chat_state, origin_image, click_state, image_input, sketcher_input,
                               image_embedding, original_size, input_size, img_caption, model_state])

        # 点击图片
        example_image.change(upload_callback, [example_image, state, text_refiner],
                             [chatbot, state, chat_state, origin_image, click_state, image_input, sketcher_input,
                              image_embedding, original_size, input_size, img_caption, model_state])


        # 文本对话；聊天
        # chat_input.submit(chat_with_points, [chat_input, click_state, chat_state, state, text_refiner, img_caption],
        #                   [chatbot, state, chat_state])
        
        # chat_input.submit(
        #     lambda x: ("", x),
        #     inputs=[chat_input],
        #     outputs=[chat_input, state],
        #     queue=False
        # ).then(
        #     chat_with_points,
        #     [chat_input, click_state, chat_state, state, text_refiner, img_caption],
        #     [chatbot, state, chat_state]
        # )


        # select coordinate;点击图像
        image_input.select(
            inference_click,
            inputs=[
                origin_image, point_prompt, click_mode, enable_wiki, language, sentiment, factuality, length,
                image_embedding, state, click_state, original_size, input_size, text_refiner,chat_state
            ],
            outputs=[chatbot, state, click_state, image_input, wiki_output,chat_state],
            show_progress=False, queue=True
        )
        # 
        submit_button_sketcher.click(
            inference_traject,
            inputs=[
                sketcher_input, enable_wiki, language, sentiment, factuality, length, image_embedding, state,
                original_size, input_size, text_refiner,chat_state
            ],
            outputs=[chatbot, state, sketcher_input, wiki_output,chat_state],
            show_progress=False, queue=True
        )

        submit_button_text.click(
                            chat_with_points,
                            [chat_input, click_state, chat_state, state, text_refiner, img_caption,length,sentiment,factuality,language],
                            [chatbot, state, chat_state]).then(lambda: "", None, chat_input).then(lambda: "",None,chat_input)
        

        return iface

# args.gradio_share
if __name__ == '__main__':
    iface = create_ui()
    iface.queue(concurrency_count=5, api_open=False, max_size=10)
    iface.launch(server_name="0.0.0.0" if not is_platform_win() else None, enable_queue=True, server_port=args.port,
                 share=True)