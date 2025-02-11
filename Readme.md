# Hemingway GRPO

A writing model that uses Guided Reward Policy Optimization (GRPO) to generate text in Hemingway's writing style, using a fork of Unsloth's Qwen2_5_(3B)_GRPO jupyter notebook.

## Overview

This project implements a reward system that trains language models to write in Hemingway's distinctive style using GRPO. The system evaluates text based on:
- Writing clarity
- Sentence structure
- Output length
- Adherence to Hemingway's style characteristics

## Key Components

- `train_hemingway.py`: Main training script with reward functions
- `hemingway.py`: Hemingway/readability scripts
- `grpo_gsm8k_reasoning.py`: @willccbb's original GRPO reward demo for reasoning
- `Qwen2_5_(3B)_GRPO_Hemingway.ipynb`: My modified unsloth jupyter notebook. WIP, doesn't converge

## Requirements

See `requirements.txt` for dependencies.

## Usage

Training and implementation details coming soon.
