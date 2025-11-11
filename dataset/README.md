# Dataset 

Benchmark datasets used to evaluate the framework for analyzing and manipulating Vision-Language Models’ safety mechanisms.

## 🗂️ Key Benchmark Datasets for evaluating LVLM performance


## 🗂️ Key Benchmark Datasets for evaluating Jailbreaking Attacks

### 1. SafeBench (Figstep)
🤗 [Zonghao2025/safebench](https://huggingface.co/datasets/Zonghao2025/safebench)        



### 2. JailBreakV-28K
🤗 [JailbreakV-28K/JailBreakV-28k](https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k)   
- The **JailBreakV-28K** dataset is a comprehensive benchmark designed to evaluate the **transferability of jailbreak attacks** from **LLMs** to **MLLMs**.
- It is built upon the **RedTeam-2K** dataset, which comprises **2,000 harmful queries** developed to identify alignment vulnerabilities in LLMs and MLLMs. The RedTeam-2K dataset aggregates queries from eight distinct sources: GPT Rewrite, Handcraft, GPT Generate, LLM Jailbreak Study, AdvBench, BeaverTails, Question Set, and Anthropic’s hh-rlhf.
- Specifically, JailBreakV-28K contains **28,000 text–image jailbreak pairs**, including **20,000 text-based LLM-transfer attacks** and **8,000 image-based MLLM attacks**.
- The JailBreakV-28K dataset incorporates five diverse jailbreak strategies.

  <table border="1" cellpadding="6" cellspacing="0">
    <thead>
      <tr>
        <th>Category</th>
        <th>Attack Type</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="3">
          <div style="background-color:white;padding:4px 6px;"><strong>LLM-Transfer Attacks</strong></div>
        </td>
        <td><strong>Logic (Cognitive Overload)</strong></td>
        <td>Exploits reasoning overload by constructing logically complex or contradictory instructions.</td>
      </tr><tr>
        <td><strong>Persuade (Persuasive Adversarial Prompts)</strong></td>
        <td>Uses manipulative or emotionally persuasive phrasing to bypass safety alignment.</td>
      </tr><tr>
        <td><strong>Template (Greedy Coordinate Gradient / Handcrafted)</strong></td>
        <td>Applies structured templates, including algorithmically optimized or handcrafted prompt patterns.</td>
      </tr><tr>
        <td rowspan="2">
          <div style="background-color:white;padding:4px 6px;"><strong>MLLM-Specific Attacks</strong></div>
        </td>
        <td><strong>FigStep</strong></td>
        <td>Multimodal jailbreak using typographic and visual perturbations to induce unsafe outputs.</td>
      </tr><tr>
        <td><strong>Query-Relevant Attack</strong></td>
        <td>Designs image–text pairs closely tied to harmful query semantics to elicit aligned violations.</td>
      </tr>
    </tbody>
  </table>

- Furthermore, JailBreakV-28K offers a broad spectrum of attack methodologies and integrates various **image types like Nature**, **Random Noise**, **Typography**, **Stable Diffusion (SD)**, **Blank**, and **SD + Typography images**.



### 3. HarmBench
🤗 [walledai/HarmBench](https://huggingface.co/datasets/walledai/HarmBench)   
On the HarmBench [GitHub page](https://github.com/centerforaisafety/HarmBench?tab=readme-ov-file), the data folder contains the following four datasets:

<table border="1" cellpadding="6" cellspacing="0">
  <thead>
    <tr>
      <th>Folder</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><div style="background-color:white;padding:4px 6px;"><strong>behavior_datasets</strong></div></td>
      <td>HarmBench behavior datasets stored as CSVs. The full set is split into text and multimodal behaviors (different formats). Validation and test splits are included. Extra example datasets (AdvBench and TDC 2023 Red Teaming Track) are available under <code>behavior_datasets/extra_behavior_datasets</code>.</td>
    </tr><tr>
      <td><div style="background-color:white;padding:4px 6px;"><strong>copyright_classifier_hashes</strong></div></td>
      <td>Hashes used by the copyright classifier for copyright-related behaviors; loaded during evaluation to detect whether model completions contain parts of books or song lyrics specified in a copyright behavior.</td>
    </tr><tr>
      <td><div style="background-color:white;padding:4px 6px;"><strong>multimodal_behavior_images</strong></div></td>
      <td>**Images** referenced by **multimodal behaviors** (the visual assets used when evaluating multimodal test cases). (See <a href="https://github.com/centerforaisafety/HarmBench/tree/main/data/multimodal_behavior_images" target="_blank">multimodal behavior images</a> for reference.) </td>
    </tr><tr>
      <td><div style="background-color:white;padding:4px 6px;"><strong>optimizer_targets</strong></div></td>
      <td>Standard target sets used by many red-teaming optimization methods (similar to targets in the <a href="https://arxiv.org/abs/2307.15043" target="_blank">GCG paper</a>). Additional target sets are provided in <code>optimizer_targets/extra_targets</code>, including sets for adversarial training, AdvBench, and model-specific custom targets.</td>
    </tr>
  </tbody>
</table>





### 4. AdvBench
🤗 [yukiyounai/AdvBench](https://huggingface.co/datasets/yukiyounai/AdvBench)   

