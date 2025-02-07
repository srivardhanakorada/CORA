# **Precise, Fast, and Low-cost Concept Erasure in Value Space: Orthogonal Complement Matters**

Paper: https://arxiv.org/abs/2412.06143

Author: Yuan Wang, Ouxiang Li, Tingting Mu, Yanbin Hao, Kuien Liu, Xiang Wang, Xiangnan He

![image.png](img/overview.png)

**Overview of our Adaptive Value Decomposer (AdaVD)** in erasing the target concept “*Snoopy*” (a) First, we token-wisely duplicate the last subject token of the target embedding encoded by the text encoder, except for[SOT]. (b) Then, the pre-processed target embedding and corresponding prompt embedding are jointly fed into CA layers within the UNet as conditions, to disentangle target semantics from the original image at each timestep. (c) In each CA layer, we perform token-wise orthogonal value decomposition with an adaptive token-wise shift. The new value is subsequently multiplied by the attention map, producing the erased output for this CA layer.

## Getting Started

### **01. Setup for experiments**

```bash
conda env create -f environment.yml
```

### 02. Image Generation with AdaVD

- **Generate with the Provided code**
    
    To evaluate our **AdaVD using the provided benchmark** and obtain quantitative performance metrics for concept erasure, please run the following commands:
    
    ```bash
    CUDA_VISIBLE_DEVICES=${gpu_id} python src/main.py \
        --erase_type ${erased_concept_type} \
        --target_concept ${erased_concept_1, erased_concept_2, ..., erased_concept_m} \
        --contents ${evaluate_concept_1, evaluate_concept_2, ..., evaluate_concept_n} \
        --mode ${sample_mode} \
        --num_samples ${img_per_prompt} --batch_size ${sample_bs} \
        --save_root ${your_save_path} \
    
    # Example:    
    CUDA_VISIBLE_DEVICES=${gpu_id} python src/main.py \
        --erase_type 'instance' \
        --target_concept 'Snoopy, Mickey, Spongebob' \
        --contents 'Snoopy, Mickey, Spongebob, Pikachu, Dog, Legislator' \
        --mode 'original, erase, retain' \
        --num_samples 10 --batch_size 10 \
        --save_root ${your_save_path} \
    ```
    
    In the commands above, you can configure the `--sample_mode` parameter to determine the sampling mode:
    
    - **`original`**: Generates images using the original Stable Diffusion model.
    - **`retain`**: Produces images after target concept erasure.
    - **`erase`**: Visualizes the erased components.
    
    Additionally, you can set the `--erase_type` parameter to evaluate AdaVD's performance in erasing different types of concepts:
    
    - **`instance`**: On Instance Erasure.
    - **`style`**: On Art Style Erasure.
    - **`celebrity`**: On Celebrity Erasure.
    
    You can also adjust the hyper-parameters `--sigmoid_a`, `--sigmoid_b`, and `--sigmoid_c` to erase implicit  concept.
    
    If you want to evaluate AdaVD with **your own benchmark** or **apply it to eliminate numerous concepts**, please run the following commands:
    
    ```bash
    CUDA_VISIBLE_DEVICES=${gpu_id} python src/main_multi.py \
        --erase_type ${your_benchmark_name} \
        --target_concept ${erased_concept_1, erased_concept_2, ..., erased_concept_m} \
        --contents 'erase, retention' \
        --mode ${sample_mode} \
        --num_samples ${img_per_prompt} --batch_size ${sample_bs} \
        --save_root ${your_save_path}
        
    # Example:  
    CUDA_VISIBLE_DEVICES=${gpu_id} python src/main_multi.py \
        --erase_type '10_celebrity' \
        --target_concept 'Angelina Jolie, Anjelica Huston, Anna Faris, Anna Kendrick, Adam Driver, Adriana Lima, Amber Heard, Amy Adams, Andrew Garfield, Anne Hathaway' \
        --contents 'erase, retention' \
        --mode 'original, retain' \
        --num_samples 10 --batch_size 10 \
        --save_root ${your_save_path}
    ```
    
    In the above command, you can configure the `erase_type` to use your benchmark. The structure of the YAML file for your benchmark should be as follows:
    
    ```yaml
    id: prompt index
    type: 'erase' or 'retention'  # Determines whether the prompt refers to the target concept or non-target concept
    text: prompt for image generation
    concept: concept corresponding to the prompt
    seed: sample seed
    
    Example: 
    id, type, text, concept, seed
    1,erase,A portrait of Adam Driver,Adam Driver,1
    ```
    
- **Generate in the notebook demo**
    
    You can also use the [`demo.ipynb`](https://github.com/Con6924/SPM/blob/main/demo.ipynb) notebook to quickly generate examples and compare the original generated image, the visualized erased component, and the image after concept erasure. This provides a clear understanding of the erasure efficacy and prior preservation performance of our AdaVD.
    

## **03. Evaluate AdaVD**

To validate the results presented in our paper, you can run the following commands:
```bash
CUDA_VISIBLE_DEVICES=${gpu_id} python src/clip_score_cal.py \
  --contents ${evaluate_concept} \
  --root_path ${root_path_of_erased_image} \
  --sub_root ${sub_root_for_evaluation} \
  --pretrained_path ${path_of_original_image}
  
# Example: 
CUDA_VISIBLE_DEVICES=${gpu_id} python src/clip_score_cal.py \
  --contents 'Van Gogh, Picasso, Monet, Andy Warhol, Caravaggio' \
  --root_path 'logs/style' \
  --sub_root 'retain' \
  --pretrained_path 'data/pretrain/style'
```