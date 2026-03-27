import os
import io
import time
import base64
import requests
import streamlit as st
from PIL import Image

# ==================== 默认配置 ====================
DEFAULT_API_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
DEFAULT_MODEL = "doubao-seedream-4-5-251128"
TARGET_SIZE = (800, 800)
API_KEY = "e352cec2-0723-4eb8-8bfa-e1b00aba13cd"
# e352cec2-0723-4eb8-8bfa-e1b00aba13cd

# ==================== 工具函数 ====================
def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def encode_bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def guess_mime_from_pil(img: Image.Image) -> str:
    fmt = (img.format or "PNG").upper()
    if fmt in ("JPG", "JPEG"):
        return "jpeg"
    if fmt == "PNG":
        return "png"
    if fmt == "WEBP":
        return "webp"
    return "png"


def load_image_from_path(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img


def load_image_from_upload(uploaded_file) -> Image.Image:
    img = Image.open(uploaded_file)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img


def resize_to_target_cover(img: Image.Image, target_size=(800, 800)) -> Image.Image:
    tw, th = target_size
    w, h = img.size
    scale = max(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return resized.crop((left, top, left + tw, top + th))


def remove_bg_and_resize(img: Image.Image, target_size=(800, 800)) -> Image.Image:
    try:
        from rembg import remove
    except ImportError as e:
        raise RuntimeError("未安装 rembg：请先 pip install rembg") from e
    cutout = remove(img)
    if cutout.mode != "RGBA":
        cutout = cutout.convert("RGBA")
    alpha = cutout.split()[3]
    bbox = alpha.getbbox()
    content = cutout.crop(bbox) if bbox else cutout
    return content.resize(target_size, Image.Resampling.LANCZOS)


def ensure_min_size_for_api(img: Image.Image, min_pixels=3686400) -> Image.Image:
    w, h = img.size
    current_pixels = w * h
    if current_pixels >= min_pixels:
        return img
    scale = (min_pixels / current_pixels) ** 0.5
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def call_image_generation_api(api_url, api_key, model, prompt, input_img, size="2K", watermark=False, timeout=120):
    mime = guess_mime_from_pil(input_img)
    image_bytes = pil_to_bytes(input_img, fmt="PNG" if mime == "png" else "JPEG")
    b64 = encode_bytes_to_base64(image_bytes)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "image": f"data:image/{mime};base64,{b64}",
        "sequential_image_generation": "disabled",
        "response_format": "url",
        "stream": False,
        "watermark": watermark,
        "size": size,
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    if "data" in data and data["data"]:
        url = data["data"][0].get("url")
        if not url:
            raise RuntimeError("API返回data但没有url字段")
        return url
    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")
    raise RuntimeError(f"未知响应格式: {str(data)[:500]}")


def download_image_to_pil(url, timeout=60):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content))
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img


def get_all_images_from_folder(folder_path, recursive=True):
    supported_ext = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif')
    image_files = []

    if recursive:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(supported_ext):
                    image_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder_path):
            if file.lower().endswith(supported_ext):
                full_path = os.path.join(folder_path, file)
                if os.path.isfile(full_path):
                    image_files.append(full_path)

    return sorted(image_files)


def get_subfolders(folder_path):
    """获取所有直接子文件夹"""
    subfolders = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            subfolders.append(item_path)
    return sorted(subfolders)


def count_images_in_folder(folder_path):
    """统计文件夹中的图片数量（不递归子文件夹）"""
    supported_ext = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif')
    count = 0
    for file in os.listdir(folder_path):
        if file.lower().endswith(supported_ext):
            full_path = os.path.join(folder_path, file)
            if os.path.isfile(full_path):
                count += 1
    return count


# ==================== Streamlit UI ====================
st.set_page_config(page_title="图生图生成工具", layout="wide")
st.title("图生图生成工具（支持多Prompt选择）")

# ==================== 侧边栏配置 ====================
with st.sidebar:
    # ========== Prompt配置（移到最顶部）==========
    st.header("📝 Prompt 配置")

    # Prompt 1 配置
    st.subheader("Prompt 1")
    prompt_1_name = st.text_input("名称", value="扣透明图800*800", key="p1_name")
    prompt_1_content = st.text_area(
        "内容",
        value="专业电商产品摄影，干净背景，影棚灯光，清晰对焦，高品质，无文字，无标识，无水印",
        height=80,
        key="p1_content"
    )

    st.divider()

    # Prompt 2 配置（支持品牌名输入）
    st.subheader("Prompt 2")
    prompt_2_name = st.text_input("名称", value="去品牌和水印", key="p2_name")

    # 品牌名输入
    brand_name = st.text_input(
        "🎯 要去除的品牌名（多个用逗号分隔）",
        value="",
        placeholder="例如：小米, Apple, Nike",
        key="brand_name",
        help="输入需要AI特别注意去除的品牌名称，用逗号分隔多个品牌"
    )

    # 构建Prompt 2内容（动态插入品牌名）
    base_prompt_2 = """请对这张电商商品详情图做无痕去除与修复（inpainting）：
只删除：所有品牌相关信息：品牌名、品牌Logo、店铺名/店铺标识、任何可识别品牌的图形或字样（出现在角标、顶部/底部条幅、商品瓶身/标签、包装上都算）。所有水印：平台水印、全屏/半透明水印、角落水印、重复纹理水印、二维码、防盗链字样等。
必须保留：图片里的商品介绍文案与参数信息（如功能卖点、尺寸规格、销量数据、标语等普通描述）全部保留，不要删除或改写，不要更换字体、不改排版、不改颜色。商品主体、背景、光影、质感、构图保持不变。
修复要求：删除后的区域用与周围一致的背景纹理/渐变/光效进行补全，边缘自然无痕，看不出被擦除或涂抹。若品牌信息与介绍文案重叠：只移除品牌部分，保留其余文字；无法无痕保留时，用相同字体风格与排版重建非品牌文字内容（不要新增营销内容）。禁止生成任何新的Logo/品牌名/水印/额外文字。
输出：生成一张"保留详情介绍文案、但无品牌Logo/品牌名/无水印"的干净详情图。"""

    # 如果有输入品牌名，添加到Prompt开头
    if brand_name.strip():
        brand_list = [b.strip() for b in brand_name.split(",") if b.strip()]
        if brand_list:
            brand_instruction = "【特别注意：必须完全去除以下品牌相关信息：" + "、".join(brand_list) + "】\n\n"
            full_prompt_2 = brand_instruction + base_prompt_2
        else:
            full_prompt_2 = base_prompt_2
    else:
        full_prompt_2 = base_prompt_2

    # 显示完整的Prompt 2（可编辑，但默认使用动态生成的）
    prompt_2_content = st.text_area(
        "内容（已自动包含品牌名提示）",
        value=full_prompt_2,
        height=120,
        key="p2_content"
    )

    st.divider()

    # ========== API配置 ==========
    st.header("⚙️ API 配置")
    api_key = st.text_input("API Key", value=os.getenv("ARK_API_KEY", API_KEY), type="password")
    api_url = st.text_input("API URL", value=os.getenv("ARK_API_URL", DEFAULT_API_URL))
    model = st.text_input("Model", value=os.getenv("ARK_MODEL", DEFAULT_MODEL))

    st.divider()

    # ========== 生成参数 ==========
    st.subheader("生成参数")
    size = st.selectbox("size", ["2K"], index=0)
    watermark = st.checkbox("生成图片是否带水印", value=False)
    request_delay = st.number_input("请求间隔(秒)", min_value=0.0, max_value=10.0, value=1.0, step=0.5)

    st.divider()

    # ========== 后处理 ==========
    st.subheader("后处理")
    postprocess = st.selectbox(
        "输出处理",
        ["rembg抠图+800x800（主体占满）", "裁剪缩放到800x800（保留场景）", "不处理"],
        index=0
    )

    st.divider()

    # ========== 批量设置 ==========
    st.subheader("批量设置")
    output_suffix = st.text_input("输出文件名后缀", value="_AI_generated")

# 创建prompt选项字典（在侧边栏配置后创建）
prompt_options = {
    prompt_1_name: prompt_1_content,
    prompt_2_name: prompt_2_content
}

st.divider()

# 主界面
tab1, tab2 = st.tabs(["单张处理", "批量处理（文件夹）"])

# ===== 单张处理标签 =====
with tab1:
    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.subheader("输入")
        input_mode = st.radio("输入方式", ["上传图片"], horizontal=True, key="single_mode")

        input_img = None
        input_name = "input.png"

        if input_mode == "上传图片":
            uploaded = st.file_uploader("上传单张图片", type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"],
                                        key="single_upload")
            if uploaded:
                input_img = load_image_from_upload(uploaded)
                input_name = uploaded.name
        else:
            path = st.text_input("本地图片路径", value="", key="single_path")
            if path and os.path.exists(path) and os.path.isfile(path):
                try:
                    input_img = load_image_from_path(path)
                    input_name = os.path.basename(path)
                except Exception as e:
                    st.error(f"读取图片失败：{e}")

        # 单张处理：选择使用哪个Prompt
        st.subheader("选择Prompt")
        selected_prompt_key = st.radio(
            "使用哪个Prompt？",
            options=list(prompt_options.keys()),
            horizontal=True,
            key="single_prompt_select"
        )
        selected_prompt = prompt_options[selected_prompt_key]

        # 显示选中的prompt内容（只读）
        st.text_area("当前使用的Prompt", value=selected_prompt, height=100, disabled=True, key="single_prompt_display")
        generate = st.button("开始生成（1张）", type="primary", use_container_width=True, disabled=(input_img is None),
                             key="single_generate")
        if input_img is not None:
            st.caption(f"输入预览：{input_name} | {input_img.size[0]}x{input_img.size[1]}")
            st.image(input_img, use_column_width=True)

    with col_out:
        st.subheader("输出")
        status_box = st.empty()
        log_box = st.empty()

        if "gen_img_bytes" not in st.session_state:
            st.session_state.gen_img_bytes = None
            st.session_state.post_img_bytes = None

        if generate and input_img:
            if not api_key:
                st.error("请填写 API Key")
            else:
                logs = []


                def log(msg):
                    logs.append(msg)
                    log_box.code("\n".join(logs), language="text")


                try:
                    status_box.info("准备中…")
                    log(f"1) 准备输入图片…")
                    log(f"   使用Prompt: {selected_prompt_key}")
                    req_img = ensure_min_size_for_api(input_img)
                    if req_img.size != input_img.size:
                        log(f"   自动放大: {input_img.size} → {req_img.size}")

                    status_box.info("调用API生成中…")
                    log("2) 调用API生成…")
                    url = call_image_generation_api(
                        api_url=api_url.strip(),
                        api_key=api_key.strip(),
                        model=model.strip(),
                        prompt=selected_prompt,
                        input_img=req_img,
                        size=size,
                        watermark=watermark,
                    )
                    log(f"   生成URL: {url[:60]}...")

                    time.sleep(float(request_delay))

                    status_box.info("下载与后处理…")
                    log("3) 下载生成图片…")
                    gen_img = download_image_to_pil(url)
                    _tmp = gen_img.copy()
                    _tmp.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS)
                    _canvas = Image.new("RGBA", TARGET_SIZE, (255, 255, 255, 0))
                    _canvas.paste(_tmp, ((TARGET_SIZE[0] - _tmp.size[0]) // 2, (TARGET_SIZE[1] - _tmp.size[1]) // 2))
                    st.session_state.gen_img_bytes = pil_to_bytes(_canvas, fmt="PNG")

                    log("4) 后处理…")
                    if postprocess == "裁剪缩放到800x800（保留场景）":
                        post_img = resize_to_target_cover(gen_img, TARGET_SIZE)
                    elif postprocess == "rembg抠图+800x800（主体占满）":
                        post_img = remove_bg_and_resize(gen_img, TARGET_SIZE)
                    else:
                        post_img = gen_img
                    st.session_state.post_img_bytes = pil_to_bytes(post_img, fmt="PNG")

                    status_box.success("完成")
                    log("完成 ✅")

                except Exception as e:
                    status_box.error(f"失败：{e}")
                    log(f"失败 ❌: {e}")

        if st.session_state.gen_img_bytes:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("AI生成原图")
                st.image(st.session_state.gen_img_bytes, use_column_width=True)
                st.download_button("下载AI生成原图", st.session_state.gen_img_bytes, f"ai_{input_name}", "image/png")
            with c2:
                st.caption("后处理结果")
                st.image(st.session_state.post_img_bytes, use_column_width=True)
                st.download_button("下载透明图", st.session_state.post_img_bytes, f"post_{input_name}", "image/png")

# ===== 批量处理标签 =====
with tab2:
    st.subheader("批量处理文件夹（根目录+子文件夹分别设置Prompt）")

    folder_path = st.text_input("输入文件夹路径", value="", key="batch_folder")

    if folder_path:
        if not os.path.exists(folder_path):
            st.error("路径不存在")
        elif not os.path.isdir(folder_path):
            st.error("这不是一个文件夹")
        else:
            # 获取根目录图片数量和子文件夹
            root_image_count = count_images_in_folder(folder_path)
            subfolders = get_subfolders(folder_path)

            # 显示统计信息
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.info(f"📁 根目录图片: {root_image_count} 张")
            with col_stat2:
                st.info(f"📂 子文件夹数量: {len(subfolders)} 个")

            if root_image_count == 0 and len(subfolders) == 0:
                st.warning("该文件夹为空，没有图片或子文件夹")
            else:
                # 为每个位置（根目录+子文件夹）分配Prompt
                st.subheader("📋 为每个目录选择Prompt")

                folder_prompt_map = {}  # 存储路径->prompt配置的映射

                # 1. 根目录配置（如果有图片）
                if root_image_count > 0:
                    with st.container():
                        st.markdown("---")
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown("**📁 [根目录-路径的当前文件夹]**")
                            st.caption(f"({root_image_count} 张图片)")
                        with col2:
                            root_prompt = st.radio(
                                "根目录使用Prompt",
                                options=list(prompt_options.keys()),
                                horizontal=True,
                                key="root_prompt_select",
                                help="根目录下的图片将使用此Prompt"
                            )
                            folder_prompt_map[folder_path] = {
                                'name': '[根目录-路径的当前文件夹]',
                                'prompt_key': root_prompt,
                                'prompt_content': prompt_options[root_prompt],
                                'image_count': root_image_count,
                                'is_root': True
                            }

                # 2. 每个子文件夹的配置
                if subfolders:
                    st.markdown("---")
                    st.markdown("**📂 子文件夹配置**")

                    for idx, subfolder in enumerate(subfolders):
                        folder_name = os.path.basename(subfolder)
                        img_count = count_images_in_folder(subfolder)

                        # 只显示有图片的文件夹，或显示所有但标注0张
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown(f"**{folder_name}/**")
                            st.caption(f"({img_count} 张图片)")
                        with col2:
                            if img_count > 0:
                                selected = st.radio(
                                    f"子文件夹Prompt_{idx}",
                                    options=list(prompt_options.keys()),
                                    horizontal=True,
                                    key=f"sub_prompt_{idx}",
                                    label_visibility="collapsed"
                                )
                                folder_prompt_map[subfolder] = {
                                    'name': folder_name,
                                    'prompt_key': selected,
                                    'prompt_content': prompt_options[selected],
                                    'image_count': img_count,
                                    'is_root': False
                                }
                            else:
                                st.caption("⚠️ 该文件夹没有图片，跳过")

                # 显示配置摘要
                if folder_prompt_map:
                    with st.expander("📊 查看配置摘要", expanded=True):
                        total_images = sum(cfg['image_count'] for cfg in folder_prompt_map.values())
                        st.write(f"**总计待处理: {total_images} 张图片**")
                        st.write("")
                        for path, config in folder_prompt_map.items():
                            icon = "📁" if config['is_root'] else "  📂"
                            st.write(
                                f"{icon} **{config['name']}** → Prompt: `{config['prompt_key']}` ({config['image_count']}张)")

                    # 输出设置
                    st.subheader("⚙️ 输出设置")
                    col1, col2 = st.columns(2)
                    with col1:
                        output_folder = st.text_input("输出文件夹路径（留空则保存在原目录旁）", value="",
                                                      key="batch_output")
                    with col2:
                        keep_structure = st.checkbox("保持原文件夹结构", value=True, key="batch_structure")

                    # 开始批量处理
                    if st.button("🚀 开始批量处理", type="primary", use_container_width=True, key="batch_start"):
                        if not api_key:
                            st.error("请填写 API Key")
                        else:
                            # 创建进度显示区域
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            log_container = st.container()
                            logs = []


                            def log(msg):
                                logs.append(msg)
                                with log_container:
                                    st.code("\n".join(logs[-30:]), language="text")


                            # 确定输出目录
                            if output_folder:
                                os.makedirs(output_folder, exist_ok=True)
                                base_output = output_folder
                            else:
                                base_output = folder_path

                            # 收集所有任务
                            all_tasks = []
                            for target_path, config in folder_prompt_map.items():
                                # 获取该目录下的图片（不递归子文件夹）
                                images = [f for f in os.listdir(target_path)
                                          if f.lower().endswith(
                                        ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif'))
                                          and os.path.isfile(os.path.join(target_path, f))]

                                for img_file in images:
                                    img_path = os.path.join(target_path, img_file)
                                    all_tasks.append({
                                        'path': img_path,
                                        'folder_path': target_path,
                                        'folder_name': config['name'],
                                        'prompt': config['prompt_content'],
                                        'prompt_key': config['prompt_key'],
                                        'is_root': config['is_root']
                                    })

                            total_tasks = len(all_tasks)
                            if total_tasks == 0:
                                st.error("没有找到可处理的图片")
                            else:
                                st.success(f"🎯 总共待处理: {total_tasks} 张图片")

                                success_count = 0
                                fail_count = 0

                                for idx, task in enumerate(all_tasks):
                                    progress = (idx + 1) / total_tasks
                                    progress_bar.progress(progress)
                                    status_text.info(
                                        f"处理中… {idx + 1}/{total_tasks} | "
                                        f"目录: {task['folder_name']} | "
                                        f"图片: {os.path.basename(task['path'])}"
                                    )

                                    try:
                                        log(f"\n[{idx + 1}/{total_tasks}] [{task['folder_name']}] {os.path.basename(task['path'])}")
                                        log(f"   Prompt: {task['prompt_key']}")

                                        # 1. 读取图片
                                        input_img = load_image_from_path(task['path'])

                                        # 2. 放大到API要求
                                        req_img = ensure_min_size_for_api(input_img)
                                        if req_img.size != input_img.size:
                                            log(f"   放大: {input_img.size} → {req_img.size}")

                                        # 3. 调用API
                                        url = call_image_generation_api(
                                            api_url=api_url.strip(),
                                            api_key=api_key.strip(),
                                            model=model.strip(),
                                            prompt=task['prompt'],
                                            input_img=req_img,
                                            size=size,
                                            watermark=watermark,
                                        )

                                        # 4. 延迟
                                        time.sleep(float(request_delay))

                                        # 5. 下载
                                        gen_img = download_image_to_pil(url)

                                        # 6. 后处理
                                        if postprocess == "裁剪缩放到800x800（保留场景）":
                                            final_img = resize_to_target_cover(gen_img, TARGET_SIZE)
                                        elif postprocess == "rembg抠图+800x800（主体占满）":
                                            final_img = remove_bg_and_resize(gen_img, TARGET_SIZE)
                                        else:
                                            final_img = gen_img

                                        # 7. 确定输出路径
                                        base_name = os.path.splitext(os.path.basename(task['path']))[0]

                                        if keep_structure:
                                            if task['is_root']:
                                                output_dir = base_output
                                            else:
                                                relative_folder = os.path.relpath(task['folder_path'], folder_path)
                                                output_dir = os.path.join(base_output, relative_folder)
                                        else:
                                            dir_name = "root" if task['is_root'] else os.path.basename(
                                                task['folder_path'])
                                            output_dir = os.path.join(base_output, dir_name)

                                        os.makedirs(output_dir, exist_ok=True)

                                        # 8. 保存后处理结果（800x800）
                                        output_path = os.path.join(output_dir, f"{base_name}{output_suffix}.png")
                                        final_img.save(output_path, "PNG")
                                        log(f"   ✅ 保存: {os.path.relpath(output_path, base_output)}")
                                        success_count += 1

                                    except Exception as e:
                                        log(f"   ❌ 失败: {str(e)[:100]}")
                                        fail_count += 1
                                        continue

                                # 完成总结
                                progress_bar.empty()
                                status_text.success(f"🎉 批量处理完成！成功: {success_count}, 失败: {fail_count}")
                                log(f"\n{'=' * 50}")
                                log(f"✅ 处理完成！")
                                log(f"总计: {total_tasks} 张")
                                log(f"成功: {success_count} 张")
                                log(f"失败: {fail_count} 张")
                                log(f"输出目录: {base_output}")
                else:
                    st.warning("没有可配置的目录（根目录和子文件夹都没有图片）")
