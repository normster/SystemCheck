# SystemCheck

<div align="center">

[[📝 Report]](https://arxiv.org/abs/2502.12197) [[📚 Dataset]](https://huggingface.co/datasets/normster/SystemCheck) [[🏁 Checkpoints]](https://huggingface.co/collections/normster/realguardrails-67ad484a279716130f624a49)

</div>

This repository contains code for our paper, [SystemCheck: A Closer Look at System Prompt Reliability](https://arxiv.org/abs/2502.12197), which studies the reliability of system prompts in large language models.

This repo includes:

* **evals**: Evaluation scripts for benchmarking models on RealGuardrails, Monkey Island stress test, S-RuLES, TensorTrust, S-IFEval, and MMLU
* **data_pipeline**: Scripts for generating synthetic training data

- `evals/`: Evaluation scripts and benchmarks
  - RealGuardrails benchmark and Monkey Island stress test
  - S-RuLES, TensorTrust, S-IFEval, and MMLU evaluations
  - See [evals/EVALS.md](evals/EVALS.md) for usage

- `data_pipeline/`: Synthetic data generation pipeline
  - User/assistant message generation
  - Tool use capabilities: search, browsing, etc.
  - See [data_pipeline/DATA_PIPELINE.md](data_pipeline/DATA_PIPELINE.md) for details

## Data

Our data is available on HuggingFace: [normster/SystemCheck](https://huggingface.co/datasets/normster/SystemCheck).

## Models

Fine-tuning was performed using the [torchllms](https://github.com/normster/torchllms) library. Trained model checkpoints are available on [HuggingFace](https://huggingface.co/collections/normster/realguardrails-67ad484a279716130f624a49).

## Citation

If you would like to cite this work, you may use the following BibTeX entry:

Authors: Norman Mu, Jonathan Lu, Michael Lavery, David Wagner

```bibtex
@misc{mu2025closerlookpromptrobustness,
      title={A Closer Look at System Prompt Robustness}, 
      author={Norman Mu and Jonathan Lu and Michael Lavery and David Wagner},
      year={2025},
      eprint={2502.12197},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.12197}, 
}
```
