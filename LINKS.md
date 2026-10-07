# Lab 21 — Submission Links

**Họ tên:** Nguyễn Văn Hưởng

**MSSV:** 2A202602743

**Ngày:** 07/10/2026

- **GitHub:** https://github.com/Huongne2405/Day21-Track3-Finetuning-Lab
- **Adapter công khai trên Hugging Face:** https://huggingface.co/Huongne2405/lab21-2A202602743-triage-lora
- **Report:** [submission/REPORT.md](submission/REPORT.md)
- **Reflection:** [submission/REFLECTION.md](submission/REFLECTION.md)
- **Kết quả:** [results/](results/)

## Thông tin adapter

Base model: `unsloth/Qwen3.5-4B`. Adapter được nộp là run `correct`.
Repo Hugging Face có `adapter_config.json` và `adapter_model.safetensors` ở thư mục gốc.

## Phán quyết

Target tăng từ 0,765 lên 0,970, nhưng regression giảm từ 0,7911 xuống 0,5222.
Verdict: **FAILED** do giảm regression vượt ngưỡng 0,02; kết quả được phân tích trong report.

## Mục thưởng đăng ký

**B5 (+2 điểm theo rubric):** adapter chính được công khai trên Hugging Face Hub và có link trong report.
