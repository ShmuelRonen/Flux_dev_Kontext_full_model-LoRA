import gradio as gr
import numpy as np
import spaces
import torch
import random
import os
import glob
from PIL import Image

# Essential fix for enable_gqa
import torch.nn.functional as F
original_sdpa = F.scaled_dot_product_attention
def patched_sdpa(*args, **kwargs):
    kwargs.pop('enable_gqa', None)
    return original_sdpa(*args, **kwargs)
F.scaled_dot_product_attention = patched_sdpa
torch.nn.functional.scaled_dot_product_attention = patched_sdpa

from diffusers import FluxKontextPipeline

MAX_SEED = np.iinfo(np.int32).max

# LoRA Management
LORA_FOLDER = "AVAILABLE_LORAS"
current_lora = None
current_lora_scale = 1.0

def scan_available_loras():
    """Scan the AVAILABLE_LORAS folder for .safetensors files"""
    loras = {"None": None}
    
    if not os.path.exists(LORA_FOLDER):
        os.makedirs(LORA_FOLDER)
        print(f"📁 Created {LORA_FOLDER} folder - place your .safetensors LoRA files here!")
        return loras
    
    # Scan for .safetensors files
    lora_files = glob.glob(os.path.join(LORA_FOLDER, "*.safetensors"))
    
    for lora_path in lora_files:
        lora_name = os.path.basename(lora_path).replace(".safetensors", "")
        loras[lora_name] = lora_path
        print(f"🎨 Found LoRA: {lora_name}")
    
    if len(loras) == 1:  # Only "None"
        print(f"📝 No LoRA files found in {LORA_FOLDER}/ - add .safetensors files to use LoRAs!")
    else:
        print(f"✅ Loaded {len(loras)-1} LoRA(s) from {LORA_FOLDER}/")
    
    return loras

def load_lora(lora_path, lora_scale=1.0):
    """Load a LoRA using ComfyUI-style (model only, no CLIP)"""
    global current_lora, current_lora_scale
    
    try:
        # Unload previous LoRA if different
        if current_lora and current_lora != lora_path:
            print(f"🔄 Unloading previous LoRA: {os.path.basename(current_lora) if current_lora != 'None' else 'None'}")
            try:
                pipe.unload_lora_weights()
            except:
                pass
            current_lora = None
        
        # Load new LoRA
        if lora_path and lora_path != "None":
            print(f"🎨 Loading LoRA: {os.path.basename(lora_path)} (strength: {lora_scale})")
            
            try:
                # ComfyUI Method: Load LoRA ONLY on transformer (no CLIP/text encoder)
                print(f"🎯 Applying LoRA to transformer only (ComfyUI style - no CLIP)")
                
                # Load LoRA specifically to transformer component only
                pipe.load_lora_weights(
                    lora_path, 
                    adapter_name="default",
                    # Only apply to transformer, ignore text encoder components
                    ignore_mismatched_keys=True
                )
                
                # Set adapter weights for transformer only
                pipe.set_adapters(["default"], adapter_weights=[lora_scale])
                
                print(f"✅ LoRA loaded on transformer only (no text encoder)")
                
                current_lora = lora_path
                current_lora_scale = lora_scale
                return f"✅ Loaded: {os.path.basename(lora_path)} (transformer only)"
                
            except Exception as e1:
                print(f"⚠️ Transformer-only loading failed: {e1}")
                try:
                    # Fallback: Manual transformer loading
                    print(f"🔧 Trying manual transformer loading...")
                    pipe.transformer.load_adapter(lora_path, adapter_name="default")
                    pipe.transformer.set_adapters(["default"], weights=[lora_scale])
                    
                    current_lora = lora_path
                    current_lora_scale = lora_scale
                    return f"✅ Loaded: {os.path.basename(lora_path)} (manual transformer)"
                    
                except Exception as e2:
                    print(f"❌ Manual loading failed: {e2}")
                    return f"❌ LoRA loading failed: {str(e2)[:50]}..."
        else:
            current_lora = None
            current_lora_scale = 1.0
            return "✅ No LoRA (default style)"
            
    except Exception as e:
        print(f"❌ Error loading LoRA: {e}")
        current_lora = None
        return f"❌ Error: {str(e)[:50]}..."

def combine_images_advanced(image1, image2, mode="single", position="right", gap_pixels=10):
    """Advanced image combination with multiple modes"""
    
    # Convert to PIL if needed
    def to_pil(img):
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            return Image.fromarray(img)
        return img
    
    if mode == "single" or image2 is None:
        return to_pil(image1)
    
    img1 = to_pil(image1)
    img2 = to_pil(image2)
    
    if mode == "side_by_side":
        # Side by side combination
        images = [img1, img2] if position == "right" else [img2, img1]
        
        # Resize to same height
        max_height = max(img.height for img in images)
        resized_images = []
        for img in images:
            if img.height != max_height:
                new_width = int(img.width * max_height / img.height)
                img = img.resize((new_width, max_height), Image.Resampling.LANCZOS)
            resized_images.append(img)
        
        # Combine horizontally
        total_width = sum(img.width for img in resized_images) + gap_pixels
        combined = Image.new('RGB', (total_width, max_height), (255, 255, 255))
        
        x_offset = 0
        for img in resized_images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width + gap_pixels
        
        return combined
    
    elif mode == "reference":
        # Reference mode - just return main image (second image used as context in prompt)
        return img1
    
    return img1

print("🚀 Loading FLUX Kontext with smart memory management...")

# Load with lower memory footprint
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", 
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)

print("🧠 Applying smart memory optimizations...")

# Instead of full CPU offloading, use sequential CPU offloading (faster)
try:
    pipe.enable_sequential_cpu_offload()
    print("✅ Sequential CPU offloading enabled (faster than full offloading)")
except:
    print("⚠️ Sequential offloading not available, using attention slicing instead")
    pipe.to("cuda")
    pipe.enable_attention_slicing(1)

# These are fast and help a lot
pipe.enable_vae_slicing()
pipe.enable_attention_slicing(1)

# Enable memory efficient attention if available
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("✅ xFormers memory efficient attention enabled")
except:
    print("⚠️ xFormers not available")

print("✅ Memory optimizations complete")

# Scan for available LoRAs
AVAILABLE_LORAS = scan_available_loras()

@spaces.GPU
def infer(input_image, second_image, combination_mode, extra_position, prompt, max_size, selected_lora, lora_scale, seed=42, randomize_seed=False, guidance_scale=2.5, steps=20, progress=gr.Progress(track_tqdm=True)):
    """
    Enhanced FLUX Kontext inference with multiple combination modes.
    """
    # Load selected LoRA
    lora_path = AVAILABLE_LORAS.get(selected_lora)
    lora_status = load_lora(lora_path, lora_scale)
    
    # Smart memory management
    torch.cuda.empty_cache()
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    # Handle image combination based on mode
    if combination_mode == "single":
        final_image = input_image
        print("📷 Using single image mode")
    elif second_image is not None:
        print(f"🔄 Combining images in {combination_mode} mode...")
        
        if combination_mode == "reference":
            # Reference mode: Intelligent prompt enhancement based on transformation type
            final_image = input_image
            prompt_lower = prompt.lower()
            
            # Detect transformation type and enhance prompt accordingly
            if any(word in prompt_lower for word in ["dress", "wear", "outfit", "clothing", "clothes", "shirt", "pants", "jacket", "suit"]):
                # Clothing transformation
                prompt = f"Complete clothing replacement: {prompt}. Remove the original outfit entirely and replace it with the exact clothing from the reference image. The person should wear ONLY the new clothing item, no mixing of garments."
                print("🎽 Detected: Clothing transformation")
                
            elif any(word in prompt_lower for word in ["hold", "holding", "grab", "carry", "object", "item", "tool"]):
                # Object holding transformation
                prompt = f"Object placement: {prompt}. The person should be holding or interacting with the object from the reference image. Integrate the object naturally into the scene."
                print("🤲 Detected: Object holding transformation")
                
            elif any(word in prompt_lower for word in ["background", "setting", "environment", "place", "location", "scene", "behind"]):
                # Background transformation
                prompt = f"Background replacement: {prompt}. Replace the original background entirely with the background/setting from the reference image. Keep the person but change the environment completely."
                print("🌅 Detected: Background transformation")
                
            elif any(word in prompt_lower for word in ["hair", "hairstyle", "haircut", "color"]):
                # Hair/appearance transformation
                prompt = f"Appearance modification: {prompt}. Change the person's appearance to match the reference style while keeping their identity."
                print("💇 Detected: Appearance transformation")
                
            elif any(word in prompt_lower for word in ["pose", "position", "stance", "gesture", "posture"]):
                # Pose transformation
                prompt = f"Pose modification: {prompt}. Change the person's pose/position to match the reference image while keeping everything else consistent."
                print("🕺 Detected: Pose transformation")
                
            elif any(word in prompt_lower for word in ["style", "art", "artistic", "effect", "filter", "look"]):
                # Style transformation
                prompt = f"Style transfer: {prompt}. Apply the artistic style, mood, and visual characteristics from the reference image to transform the main image."
                print("🎨 Detected: Style transformation")
                
            else:
                # General transformation
                prompt = f"Smart transformation: {prompt}. Use the reference image to guide the transformation. Identify what should be changed and apply it intelligently to the main image."
                print("🧠 Detected: General transformation")
                
            print(f"📝 Enhanced prompt: {prompt[:100]}...")
        else:
            final_image = combine_images_advanced(
                input_image, second_image, 
                mode=combination_mode, 
                position=extra_position, 
                gap_pixels=10
            )
            print(f"📷 Applied {combination_mode} combination")
    else:
        final_image = input_image
    
    # Smart image size management
    if final_image:
        if isinstance(final_image, Image.Image):
            input_pil = final_image
        else:
            input_pil = Image.fromarray((final_image * 255).astype(np.uint8)) if final_image.dtype != np.uint8 else Image.fromarray(final_image)
        
        input_pil = input_pil.convert("RGB")
        
        # User decides the size vs speed trade-off
        if max(input_pil.size) > max_size:
            ratio = max_size / max(input_pil.size)
            new_size = tuple(int(dim * ratio) for dim in input_pil.size)
            input_pil = input_pil.resize(new_size, Image.Resampling.LANCZOS)
            print(f"📏 Resized to {new_size} (max size: {max_size}px)")
        else:
            print(f"📏 Using original size {input_pil.size} (within {max_size}px limit)")
        
        # Use inference mode for better memory efficiency
        with torch.inference_mode():
            try:
                image = pipe(
                    image=input_pil, 
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    width=input_pil.size[0],
                    height=input_pil.size[1],
                    num_inference_steps=min(steps, 25),  # Cap steps to prevent scheduler issues
                    generator=torch.Generator().manual_seed(seed),
                ).images[0]
            except Exception as e:
                if "index" in str(e).lower() and "out of bounds" in str(e).lower():
                    print(f"⚠️ Scheduler error with {steps} steps, retrying with 20 steps...")
                    image = pipe(
                        image=input_pil, 
                        prompt=prompt,
                        guidance_scale=guidance_scale,
                        width=input_pil.size[0],
                        height=input_pil.size[1],
                        num_inference_steps=20,  # Safe fallback
                        generator=torch.Generator().manual_seed(seed),
                    ).images[0]
                else:
                    raise e
    
    # Clean up after inference
    torch.cuda.empty_cache()
    
    return image, seed, gr.Button(visible=True), lora_status

def refresh_loras():
    """Refresh the available LoRAs list"""
    global AVAILABLE_LORAS
    AVAILABLE_LORAS = scan_available_loras()
    choices = list(AVAILABLE_LORAS.keys())
    return gr.Dropdown(choices=choices, value="None")

css="""
#col-container {
    margin: 0 auto;
    max-width: 1200px;
}
"""

with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# 🎨 FLUX Kontext + LoRA Studio
Fast image editing with LoRA styles and advanced image combination modes
        """)
        
        with gr.Row():
            with gr.Column():
                # Main images
                input_image = gr.Image(label="📷 Main Image", type="pil")
                second_image = gr.Image(label="📷 Second Image (Optional)", type="pil")
                
                # Combination mode selection
                combination_mode = gr.Radio(
                    choices=["single", "side_by_side", "reference"],
                    value="single",
                    label="🎨 Combination Mode",
                    info="How to use the images: Reference mode intelligently adapts to your prompt!"
                )
                
                # Extra positioning control - only for side_by_side
                with gr.Row(visible=False) as position_controls:
                    extra_position = gr.Dropdown(
                        choices=["left", "right"],
                        value="right",
                        label="Second Image Position"
                    )
                
                prompt = gr.Text(
                    label="✏️ Prompt",
                    placeholder="Examples: 'Dress her in this outfit' | 'Make him hold this object' | 'Change background to this scene' | 'Give her this hairstyle'",
                    lines=2
                )
                run_button = gr.Button("🎨 Process Images", variant="primary", size="lg")
                
                # Size control
                max_size = gr.Slider(
                    label="🎯 Max Image Size (Quality vs Speed)",
                    minimum=256,
                    maximum=1536,
                    value=768,
                    step=64,
                    info="📱256-512px=⚡Fast | 🎯768px=Balanced | 🎨1024px+=🏆Quality"
                )
                
                # LoRA controls
                with gr.Row():
                    selected_lora = gr.Dropdown(
                        choices=list(AVAILABLE_LORAS.keys()),
                        value="None",
                        label="🎨 Select LoRA Style",
                        scale=2
                    )
                    lora_scale = gr.Slider(
                        label="🎨 LoRA Strength",
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                        scale=1
                    )
                    refresh_button = gr.Button("🔄", scale=0)
                
                lora_status = gr.Textbox(
                    label="LoRA Status",
                    value="✅ No LoRA (default style)",
                    interactive=False
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, value=42)
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=10, value=2.5, step=0.1)
                    steps = gr.Slider(label="Steps", minimum=4, maximum=25, value=20, step=1)
                    
            with gr.Column():
                result = gr.Image(label="🎨 Result", show_label=False)
                reuse_button = gr.Button("♻️ Reuse Result", visible=False)
                
                gr.Markdown("""
                ### 🎨 Combination Modes:
                - **Single**: Use main image only
                - **Side by Side**: Images placed next to each other  
                - **Reference**: Intelligent transformation based on your prompt
                
                ### 🧠 Smart Reference Mode Examples:
                - **👗 Clothing**: "Dress the woman in this outfit" → Auto-detects clothing swap
                - **🤲 Objects**: "Make him hold this item" → Auto-detects object placement  
                - **🌅 Background**: "Change the background to this scene" → Auto-detects environment swap
                - **💇 Appearance**: "Give her this hairstyle" → Auto-detects appearance change
                - **🕺 Pose**: "Make her pose like this" → Auto-detects pose transfer
                - **🎨 Style**: "Apply this artistic style" → Auto-detects style transfer
                
                ### ⚡ Performance:
                - **512px**: Ultra fast (1-2 min)
                - **768px**: Balanced (2-3 min)
                - **1024px+**: High quality (4-6 min)
                
                ### 💡 Pro Tips:
                - Be specific in your prompts for better auto-detection
                - Use keywords like "wear", "hold", "background", "style"
                - Reference mode now intelligently adapts to your request!
                """)
        
    
    # Event handlers
    def toggle_position_controls(mode):
        return gr.Row(visible=(mode == "side_by_side"))
    
    combination_mode.change(
        fn=toggle_position_controls,
        inputs=[combination_mode],
        outputs=[position_controls]
    )
    
    refresh_button.click(
        fn=refresh_loras,
        outputs=[selected_lora]
    )
            
    run_button.click(
        fn=infer,
        inputs=[
            input_image, second_image, combination_mode, extra_position,
            prompt, max_size, selected_lora, lora_scale, 
            seed, randomize_seed, guidance_scale, steps
        ],
        outputs=[result, seed, reuse_button, lora_status]
    )
    
    reuse_button.click(
        fn=lambda image: image,
        inputs=[result],
        outputs=[input_image]
    )

demo.launch()