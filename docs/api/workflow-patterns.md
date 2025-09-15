# Workflow Patterns and Use Cases

This document provides comprehensive examples of common ComfyUI workflow patterns and real-world use cases for the Serverless API.

## Table of Contents

- [Basic Patterns](#basic-patterns)
- [Advanced Workflows](#advanced-workflows)
- [Production Use Cases](#production-use-cases)
- [Integration Patterns](#integration-patterns)
- [Performance Optimization](#performance-optimization)
- [Error Handling Patterns](#error-handling-patterns)

## Basic Patterns

### 1. Text-to-Image Generation

The most common workflow pattern for generating images from text descriptions.

```json
{
  "workflow": {
    "nodes": {
      "1": {
        "id": "1",
        "class_type": "CheckpointLoaderSimple",
        "inputs": [
          {
            "name": "ckpt_name",
            "type": "STRING",
            "value": "v1-5-pruned-emaonly.ckpt",
            "required": true
          }
        ]
      },
      "2": {
        "id": "2",
        "class_type": "CLIPTextEncode",
        "inputs": [
          {
            "name": "text",
            "type": "STRING",
            "value": "beautiful landscape, mountains, lake, sunset, masterpiece, high quality",
            "required": true
          },
          {
            "name": "clip",
            "type": "CLIP",
            "value": ["1", 1],
            "required": true
          }
        ]
      },
      "3": {
        "id": "3",
        "class_type": "CLIPTextEncode",
        "inputs": [
          {
            "name": "text",
            "type": "STRING",
            "value": "blurry, low quality, distorted, ugly",
            "required": true
          },
          {
            "name": "clip",
            "type": "CLIP",
            "value": ["1", 1],
            "required": true
          }
        ]
      },
      "4": {
        "id": "4",
        "class_type": "EmptyLatentImage",
        "inputs": [
          {
            "name": "width",
            "type": "INT",
            "value": 512,
            "required": true
          },
          {
            "name": "height",
            "type": "INT",
            "value": 512,
            "required": true
          },
          {
            "name": "batch_size",
            "type": "INT",
            "value": 1,
            "required": true
          }
        ]
      },
      "5": {
        "id": "5",
        "class_type": "KSampler",
        "inputs": [
          {
            "name": "seed",
            "type": "INT",
            "value": 42,
            "required": true
          },
          {
            "name": "steps",
            "type": "INT",
            "value": 20,
            "required": true
          },
          {
            "name": "cfg",
            "type": "FLOAT",
            "value": 7.0,
            "required": true
          },
          {
            "name": "sampler_name",
            "type": "STRING",
            "value": "euler",
            "required": true
          },
          {
            "name": "scheduler",
            "type": "STRING",
            "value": "normal",
            "required": true
          },
          {
            "name": "denoise",
            "type": "FLOAT",
            "value": 1.0,
            "required": true
          },
          {
            "name": "model",
            "type": "MODEL",
            "value": ["1", 0],
            "required": true
          },
          {
            "name": "positive",
            "type": "CONDITIONING",
            "value": ["2", 0],
            "required": true
          },
          {
            "name": "negative",
            "type": "CONDITIONING",
            "value": ["3", 0],
            "required": true
          },
          {
            "name": "latent_image",
            "type": "LATENT",
            "value": ["4", 0],
            "required": true
          }
        ]
      },
      "6": {
        "id": "6",
        "class_type": "VAEDecode",
        "inputs": [
          {
            "name": "samples",
            "type": "LATENT",
            "value": ["5", 0],
            "required": true
          },
          {
            "name": "vae",
            "type": "VAE",
            "value": ["1", 2],
            "required": true
          }
        ]
      },
      "7": {
        "id": "7",
        "class_type": "SaveImage",
        "inputs": [
          {
            "name": "images",
            "type": "IMAGE",
            "value": ["6", 0],
            "required": true
          },
          {
            "name": "filename_prefix",
            "type": "STRING",
            "value": "txt2img_output",
            "required": false
          }
        ]
      }
    },
    "metadata": {
      "description": "Basic text-to-image generation workflow",
      "use_case": "Content creation, concept art, social media",
      "version": "1.0"
    }
  }
}
```

### 2. Image-to-Image Transformation

Transform existing images while preserving structure or style.

```json
{
  "workflow": {
    "nodes": {
      "1": {
        "id": "1",
        "class_type": "LoadImage",
        "inputs": [
          {
            "name": "image",
            "type": "STRING",
            "value": "file_input_image_123",
            "required": true
          }
        ]
      },
      "2": {
        "id": "2",
        "class_type": "CheckpointLoaderSimple",
        "inputs": [
          {
            "name": "ckpt_name",
            "type": "STRING",
            "value": "v1-5-pruned-emaonly.ckpt",
            "required": true
          }
        ]
      },
      "3": {
        "id": "3",
        "class_type": "CLIPTextEncode",
        "inputs": [
          {
            "name": "text",
            "type": "STRING",
            "value": "oil painting style, artistic, vibrant colors",
            "required": true
          },
          {
            "name": "clip",
            "type": "CLIP",
            "value": ["2", 1],
            "required": true
          }
        ]
      },
      "4": {
        "id": "4",
        "class_type": "CLIPTextEncode",
        "inputs": [
          {
            "name": "text",
            "type": "STRING",
            "value": "blurry, low quality",
            "required": true
          },
          {
            "name": "clip",
            "type": "CLIP",
            "value": ["2", 1],
            "required": true
          }
        ]
      },
      "5": {
        "id": "5",
        "class_type": "VAEEncode",
        "inputs": [
          {
            "name": "pixels",
            "type": "IMAGE",
            "value": ["1", 0],
            "required": true
          },
          {
            "name": "vae",
            "type": "VAE",
            "value": ["2", 2],
            "required": true
          }
        ]
      },
      "6": {
        "id": "6",
        "class_type": "KSampler",
        "inputs": [
          {
            "name": "seed",
            "type": "INT",
            "value": 42,
            "required": true
          },
          {
            "name": "steps",
            "type": "INT",
            "value": 20,
            "required": true
          },
          {
            "name": "cfg",
            "type": "FLOAT",
            "value": 7.0,
            "required": true
          },
          {
            "name": "sampler_name",
            "type": "STRING",
            "value": "euler",
            "required": true
          },
          {
            "name": "scheduler",
            "type": "STRING",
            "value": "normal",
            "required": true
          },
          {
            "name": "denoise",
            "type": "FLOAT",
            "value": 0.75,
            "required": true
          },
          {
            "name": "model",
            "type": "MODEL",
            "value": ["2", 0],
            "required": true
          },
          {
            "name": "positive",
            "type": "CONDITIONING",
            "value": ["3", 0],
            "required": true
          },
          {
            "name": "negative",
            "type": "CONDITIONING",
            "value": ["4", 0],
            "required": true
          },
          {
            "name": "latent_image",
            "type": "LATENT",
            "value": ["5", 0],
            "required": true
          }
        ]
      },
      "7": {
        "id": "7",
        "class_type": "VAEDecode",
        "inputs": [
          {
            "name": "samples",
            "type": "LATENT",
            "value": ["6", 0],
            "required": true
          },
          {
            "name": "vae",
            "type": "VAE",
            "value": ["2", 2],
            "required": true
          }
        ]
      },
      "8": {
        "id": "8",
        "class_type": "SaveImage",
        "inputs": [
          {
            "name": "images",
            "type": "IMAGE",
            "value": ["7", 0],
            "required": true
          },
          {
            "name": "filename_prefix",
            "type": "STRING",
            "value": "img2img_output",
            "required": false
          }
        ]
      }
    },
    "metadata": {
      "description": "Image-to-image transformation workflow",
      "use_case": "Style transfer, image enhancement, artistic filters",
      "denoise_strength": 0.75,
      "version": "1.0"
    }
  }
}
```

### 3. Batch Image Generation

Generate multiple variations with different parameters.

```json
{
  "workflow": {
    "nodes": {
      "1": {
        "id": "1",
        "class_type": "CheckpointLoaderSimple",
        "inputs": [
          {
            "name": "ckpt_name",
            "type": "STRING",
            "value": "v1-5-pruned-emaonly.ckpt",
            "required": true
          }
        ]
      },
      "2": {
        "id": "2",
        "class_type": "CLIPTextEncode",
        "inputs": [
          {
            "name": "text",
            "type": "STRING",
            "value": "portrait of a person, professional photography, studio lighting",
            "required": true
          },
          {
            "name": "clip",
            "type": "CLIP",
            "value": ["1", 1],
            "required": true
          }
        ]
      },
      "3": {
        "id": "3",
        "class_type": "CLIPTextEncode",
        "inputs": [
          {
            "name": "text",
            "type": "STRING",
            "value": "blurry, low quality, distorted",
            "required": true
          },
          {
            "name": "clip",
            "type": "CLIP",
            "value": ["1", 1],
            "required": true
          }
        ]
      },
      "4": {
        "id": "4",
        "class_type": "EmptyLatentImage",
        "inputs": [
          {
            "name": "width",
            "type": "INT",
            "value": 512,
            "required": true
          },
          {
            "name": "height",
            "type": "INT",
            "value": 768,
            "required": true
          },
          {
            "name": "batch_size",
            "type": "INT",
            "value": 4,
            "required": true
          }
        ]
      },
      "5": {
        "id": "5",
        "class_type": "KSampler",
        "inputs": [
          {
            "name": "seed",
            "type": "INT",
            "value": -1,
            "required": true
          },
          {
            "name": "steps",
            "type": "INT",
            "value": 25,
            "required": true
          },
          {
            "name": "cfg",
            "type": "FLOAT",
            "value": 7.5,
            "required": true
          },
          {
            "name": "sampler_name",
            "type": "STRING",
            "value": "dpmpp_2m",
            "required": true
          },
          {
            "name": "scheduler",
            "type": "STRING",
            "value": "karras",
            "required": true
          },
          {
            "name": "denoise",
            "type": "FLOAT",
            "value": 1.0,
            "required": true
          },
          {
            "name": "model",
            "type": "MODEL",
            "value": ["1", 0],
            "required": true
          },
          {
            "name": "positive",
            "type": "CONDITIONING",
            "value": ["2", 0],
            "required": true
          },
          {
            "name": "negative",
            "type": "CONDITIONING",
            "value": ["3", 0],
            "required": true
          },
          {
            "name": "latent_image",
            "type": "LATENT",
            "value": ["4", 0],
            "required": true
          }
        ]
      },
      "6": {
        "id": "6",
        "class_type": "VAEDecode",
        "inputs": [
          {
            "name": "samples",
            "type": "LATENT",
            "value": ["5", 0],
            "required": true
          },
          {
            "name": "vae",
            "type": "VAE",
            "value": ["1", 2],
            "required": true
          }
        ]
      },
      "7": {
        "id": "7",
        "class_type": "SaveImage",
        "inputs": [
          {
            "name": "images",
            "type": "IMAGE",
            "value": ["6", 0],
            "required": true
          },
          {
            "name": "filename_prefix",
            "type": "STRING",
            "value": "batch_portraits",
            "required": false
          }
        ]
      }
    },
    "metadata": {
      "description": "Batch generation of portrait variations",
      "use_case": "A/B testing, multiple options, portfolio generation",
      "batch_size": 4,
      "version": "1.0"
    }
  }
}
```

## Advanced Workflows

### 1. LoRA-Enhanced Generation

Using LoRA (Low-Rank Adaptation) models for style-specific generation.

```json
{
  "workflow": {
    "nodes": {
      "1": {
        "id": "1",
        "class_type": "CheckpointLoaderSimple",
        "inputs": [
          {
            "name": "ckpt_name",
            "type": "STRING",
            "value": "v1-5-pruned-emaonly.ckpt",
            "required": true
          }
        ]
      },
      "2": {
        "id": "2",
        "class_type": "LoraLoader",
        "inputs": [
          {
            "name": "model",
            "type": "MODEL",
            "value": ["1", 0],
            "required": true
          },
          {
            "name": "clip",
            "type": "CLIP",
            "value": ["1", 1],
            "required": true
          },
          {
            "name": "lora_name",
            "type": "STRING",
            "value": "detail_tweaker_lora.safetensors",
            "required": true
          },
          {
            "name": "strength_model",
            "type": "FLOAT",
            "value": 0.8,
            "required": true
          },
          {
            "name": "strength_clip",
            "type": "FLOAT",
            "value": 0.8,
            "required": true
          }
        ]
      },
      "3": {
        "id": "3",
        "class_type": "CLIPTextEncode",
        "inputs": [
          {
            "name": "text",
            "type": "STRING",
            "value": "highly detailed fantasy castle, intricate architecture, dramatic lighting, <lora:detail_tweaker:0.8>",
            "required": true
          },
          {
            "name": "clip",
            "type": "CLIP",
            "value": ["2", 1],
            "required": true
          }
        ]
      },
      "4": {
        "id": "4",
        "class_type": "CLIPTextEncode",
        "inputs": [
          {
            "name": "text",
            "type": "STRING",
            "value": "blurry, low quality, simple, plain",
            "required": true
          },
          {
            "name": "clip",
            "type": "CLIP",
            "value": ["2", 1],
            "required": true
          }
        ]
      },
      "5": {
        "id": "5",
        "class_type": "EmptyLatentImage",
        "inputs": [
          {
            "name": "width",
            "type": "INT",
            "value": 768,
            "required": true
          },
          {
            "name": "height",
            "type": "INT",
            "value": 768,
            "required": true
          },
          {
            "name": "batch_size",
            "type": "INT",
            "value": 1,
            "required": true
          }
        ]
      },
      "6": {
        "id": "6",
        "class_type": "KSampler",
        "inputs": [
          {
            "name": "seed",
            "type": "INT",
            "value": 12345,
            "required": true
          },
          {
            "name": "steps",
            "type": "INT",
            "value": 30,
            "required": true
          },
          {
            "name": "cfg",
            "type": "FLOAT",
            "value": 8.0,
            "required": true
          },
          {
            "name": "sampler_name",
            "type": "STRING",
            "value": "dpmpp_2m",
            "required": true
          },
          {
            "name": "scheduler",
            "type": "STRING",
            "value": "karras",
            "required": true
          },
          {
            "name": "denoise",
            "type": "FLOAT",
            "value": 1.0,
            "required": true
          },
          {
            "name": "model",
            "type": "MODEL",
            "value": ["2", 0],
            "required": true
          },
          {
            "name": "positive",
            "type": "CONDITIONING",
            "value": ["3", 0],
            "required": true
          },
          {
            "name": "negative",
            "type": "CONDITIONING",
            "value": ["4", 0],
            "required": true
          },
          {
            "name": "latent_image",
            "type": "LATENT",
            "value": ["5", 0],
            "required": true
          }
        ]
      },
      "7": {
        "id": "7",
        "class_type": "VAEDecode",
        "inputs": [
          {
            "name": "samples",
            "type": "LATENT",
            "value": ["6", 0],
            "required": true
          },
          {
            "name": "vae",
            "type": "VAE",
            "value": ["1", 2],
            "required": true
          }
        ]
      },
      "8": {
        "id": "8",
        "class_type": "SaveImage",
        "inputs": [
          {
            "name": "images",
            "type": "IMAGE",
            "value": ["7", 0],
            "required": true
          },
          {
            "name": "filename_prefix",
            "type": "STRING",
            "value": "lora_enhanced",
            "required": false
          }
        ]
      }
    },
    "metadata": {
      "description": "LoRA-enhanced high-detail generation",
      "use_case": "High-quality artwork, detailed illustrations, professional content",
      "lora_models": ["detail_tweaker_lora.safetensors"],
      "version": "1.0"
    }
  }
}
```

### 2. ControlNet-Guided Generation

Using ControlNet for precise structural control.

```json
{
  "workflow": {
    "nodes": {
      "1": {
        "id": "1",
        "class_type": "LoadImage",
        "inputs": [
          {
            "name": "image",
            "type": "STRING",
            "value": "file_controlnet_input_456",
            "required": true
          }
        ]
      },
      "2": {
        "id": "2",
        "class_type": "CheckpointLoaderSimple",
        "inputs": [
          {
            "name": "ckpt_name",
            "type": "STRING",
            "value": "v1-5-pruned-emaonly.ckpt",
            "required": true
          }
        ]
      },
      "3": {
        "id": "3",
        "class_type": "ControlNetLoader",
        "inputs": [
          {
            "name": "control_net_name",
            "type": "STRING",
            "value": "control_v11p_sd15_canny.pth",
            "required": true
          }
        ]
      },
      "4": {
        "id": "4",
        "class_type": "CannyEdgePreprocessor",
        "inputs": [
          {
            "name": "image",
            "type": "IMAGE",
            "value": ["1", 0],
            "required": true
          },
          {
            "name": "low_threshold",
            "type": "INT",
            "value": 100,
            "required": true
          },
          {
            "name": "high_threshold",
            "type": "INT",
            "value": 200,
            "required": true
          }
        ]
      },
      "5": {
        "id": "5",
        "class_type": "ControlNetApply",
        "inputs": [
          {
            "name": "conditioning",
            "type": "CONDITIONING",
            "value": ["7", 0],
            "required": true
          },
          {
            "name": "control_net",
            "type": "CONTROL_NET",
            "value": ["3", 0],
            "required": true
          },
          {
            "name": "image",
            "type": "IMAGE",
            "value": ["4", 0],
            "required": true
          },
          {
            "name": "strength",
            "type": "FLOAT",
            "value": 1.0,
            "required": true
          }
        ]
      },
      "6": {
        "id": "6",
        "class_type": "CLIPTextEncode",
        "inputs": [
          {
            "name": "text",
            "type": "STRING",
            "value": "beautiful anime character, detailed art style, vibrant colors",
            "required": true
          },
          {
            "name": "clip",
            "type": "CLIP",
            "value": ["2", 1],
            "required": true
          }
        ]
      },
      "7": {
        "id": "7",
        "class_type": "CLIPTextEncode",
        "inputs": [
          {
            "name": "text",
            "type": "STRING",
            "value": "blurry, low quality, distorted",
            "required": true
          },
          {
            "name": "clip",
            "type": "CLIP",
            "value": ["2", 1],
            "required": true
          }
        ]
      },
      "8": {
        "id": "8",
        "class_type": "EmptyLatentImage",
        "inputs": [
          {
            "name": "width",
            "type": "INT",
            "value": 512,
            "required": true
          },
          {
            "name": "height",
            "type": "INT",
            "value": 512,
            "required": true
          },
          {
            "name": "batch_size",
            "type": "INT",
            "value": 1,
            "required": true
          }
        ]
      },
      "9": {
        "id": "9",
        "class_type": "KSampler",
        "inputs": [
          {
            "name": "seed",
            "type": "INT",
            "value": 789,
            "required": true
          },
          {
            "name": "steps",
            "type": "INT",
            "value": 25,
            "required": true
          },
          {
            "name": "cfg",
            "type": "FLOAT",
            "value": 7.5,
            "required": true
          },
          {
            "name": "sampler_name",
            "type": "STRING",
            "value": "euler_a",
            "required": true
          },
          {
            "name": "scheduler",
            "type": "STRING",
            "value": "normal",
            "required": true
          },
          {
            "name": "denoise",
            "type": "FLOAT",
            "value": 1.0,
            "required": true
          },
          {
            "name": "model",
            "type": "MODEL",
            "value": ["2", 0],
            "required": true
          },
          {
            "name": "positive",
            "type": "CONDITIONING",
            "value": ["5", 0],
            "required": true
          },
          {
            "name": "negative",
            "type": "CONDITIONING",
            "value": ["7", 0],
            "required": true
          },
          {
            "name": "latent_image",
            "type": "LATENT",
            "value": ["8", 0],
            "required": true
          }
        ]
      },
      "10": {
        "id": "10",
        "class_type": "VAEDecode",
        "inputs": [
          {
            "name": "samples",
            "type": "LATENT",
            "value": ["9", 0],
            "required": true
          },
          {
            "name": "vae",
            "type": "VAE",
            "value": ["2", 2],
            "required": true
          }
        ]
      },
      "11": {
        "id": "11",
        "class_type": "SaveImage",
        "inputs": [
          {
            "name": "images",
            "type": "IMAGE",
            "value": ["10", 0],
            "required": true
          },
          {
            "name": "filename_prefix",
            "type": "STRING",
            "value": "controlnet_guided",
            "required": false
          }
        ]
      }
    },
    "metadata": {
      "description": "ControlNet Canny edge-guided generation",
      "use_case": "Precise structural control, pose transfer, architectural design",
      "controlnet_type": "canny",
      "version": "1.0"
    }
  }
}
```

### 3. Multi-Stage Upscaling Workflow

High-resolution image generation using multiple upscaling stages.

```json
{
  "workflow": {
    "nodes": {
      "1": {
        "id": "1",
        "class_type": "CheckpointLoaderSimple",
        "inputs": [
          {
            "name": "ckpt_name",
            "type": "STRING",
            "value": "v1-5-pruned-emaonly.ckpt",
            "required": true
          }
        ]
      },
      "2": {
        "id": "2",
        "class_type": "CLIPTextEncode",
        "inputs": [
          {
            "name": "text",
            "type": "STRING",
            "value": "highly detailed landscape photography, 8k resolution, ultra sharp",
            "required": true
          },
          {
            "name": "clip",
            "type": "CLIP",
            "value": ["1", 1],
            "required": true
          }
        ]
      },
      "3": {
        "id": "3",
        "class_type": "CLIPTextEncode",
        "inputs": [
          {
            "name": "text",
            "type": "STRING",
            "value": "blurry, low resolution, pixelated",
            "required": true
          },
          {
            "name": "clip",
            "type": "CLIP",
            "value": ["1", 1],
            "required": true
          }
        ]
      },
      "4": {
        "id": "4",
        "class_type": "EmptyLatentImage",
        "inputs": [
          {
            "name": "width",
            "type": "INT",
            "value": 512,
            "required": true
          },
          {
            "name": "height",
            "type": "INT",
            "value": 512,
            "required": true
          },
          {
            "name": "batch_size",
            "type": "INT",
            "value": 1,
            "required": true
          }
        ]
      },
      "5": {
        "id": "5",
        "class_type": "KSampler",
        "inputs": [
          {
            "name": "seed",
            "type": "INT",
            "value": 456,
            "required": true
          },
          {
            "name": "steps",
            "type": "INT",
            "value": 25,
            "required": true
          },
          {
            "name": "cfg",
            "type": "FLOAT",
            "value": 7.0,
            "required": true
          },
          {
            "name": "sampler_name",
            "type": "STRING",
            "value": "dpmpp_2m",
            "required": true
          },
          {
            "name": "scheduler",
            "type": "STRING",
            "value": "karras",
            "required": true
          },
          {
            "name": "denoise",
            "type": "FLOAT",
            "value": 1.0,
            "required": true
          },
          {
            "name": "model",
            "type": "MODEL",
            "value": ["1", 0],
            "required": true
          },
          {
            "name": "positive",
            "type": "CONDITIONING",
            "value": ["2", 0],
            "required": true
          },
          {
            "name": "negative",
            "type": "CONDITIONING",
            "value": ["3", 0],
            "required": true
          },
          {
            "name": "latent_image",
            "type": "LATENT",
            "value": ["4", 0],
            "required": true
          }
        ]
      },
      "6": {
        "id": "6",
        "class_type": "VAEDecode",
        "inputs": [
          {
            "name": "samples",
            "type": "LATENT",
            "value": ["5", 0],
            "required": true
          },
          {
            "name": "vae",
            "type": "VAE",
            "value": ["1", 2],
            "required": true
          }
        ]
      },
      "7": {
        "id": "7",
        "class_type": "UpscaleModelLoader",
        "inputs": [
          {
            "name": "model_name",
            "type": "STRING",
            "value": "RealESRGAN_x4plus.pth",
            "required": true
          }
        ]
      },
      "8": {
        "id": "8",
        "class_type": "ImageUpscaleWithModel",
        "inputs": [
          {
            "name": "upscale_model",
            "type": "UPSCALE_MODEL",
            "value": ["7", 0],
            "required": true
          },
          {
            "name": "image",
            "type": "IMAGE",
            "value": ["6", 0],
            "required": true
          }
        ]
      },
      "9": {
        "id": "9",
        "class_type": "VAEEncode",
        "inputs": [
          {
            "name": "pixels",
            "type": "IMAGE",
            "value": ["8", 0],
            "required": true
          },
          {
            "name": "vae",
            "type": "VAE",
            "value": ["1", 2],
            "required": true
          }
        ]
      },
      "10": {
        "id": "10",
        "class_type": "KSampler",
        "inputs": [
          {
            "name": "seed",
            "type": "INT",
            "value": 456,
            "required": true
          },
          {
            "name": "steps",
            "type": "INT",
            "value": 15,
            "required": true
          },
          {
            "name": "cfg",
            "type": "FLOAT",
            "value": 7.0,
            "required": true
          },
          {
            "name": "sampler_name",
            "type": "STRING",
            "value": "dpmpp_2m",
            "required": true
          },
          {
            "name": "scheduler",
            "type": "STRING",
            "value": "karras",
            "required": true
          },
          {
            "name": "denoise",
            "type": "FLOAT",
            "value": 0.4,
            "required": true
          },
          {
            "name": "model",
            "type": "MODEL",
            "value": ["1", 0],
            "required": true
          },
          {
            "name": "positive",
            "type": "CONDITIONING",
            "value": ["2", 0],
            "required": true
          },
          {
            "name": "negative",
            "type": "CONDITIONING",
            "value": ["3", 0],
            "required": true
          },
          {
            "name": "latent_image",
            "type": "LATENT",
            "value": ["9", 0],
            "required": true
          }
        ]
      },
      "11": {
        "id": "11",
        "class_type": "VAEDecode",
        "inputs": [
          {
            "name": "samples",
            "type": "LATENT",
            "value": ["10", 0],
            "required": true
          },
          {
            "name": "vae",
            "type": "VAE",
            "value": ["1", 2],
            "required": true
          }
        ]
      },
      "12": {
        "id": "12",
        "class_type": "SaveImage",
        "inputs": [
          {
            "name": "images",
            "type": "IMAGE",
            "value": ["11", 0],
            "required": true
          },
          {
            "name": "filename_prefix",
            "type": "STRING",
            "value": "upscaled_hires",
            "required": false
          }
        ]
      }
    },
    "metadata": {
      "description": "Multi-stage upscaling for high-resolution output",
      "use_case": "Print-quality images, detailed artwork, professional photography",
      "final_resolution": "2048x2048",
      "version": "1.0"
    }
  }
}
```

## Production Use Cases

### 1. E-commerce Product Visualization

Workflow for generating product images with consistent styling.

```python
# E-commerce product image generation template
def create_product_workflow(product_description, style="clean modern", background="white"):
    return {
        "workflow": {
            "nodes": {
                "1": {
                    "id": "1",
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": [{"name": "ckpt_name", "type": "STRING", "value": "realistic_vision_v51.safetensors", "required": True}]
                },
                "2": {
                    "id": "2",
                    "class_type": "CLIPTextEncode",
                    "inputs": [
                        {"name": "text", "type": "STRING", "value": f"product photography, {product_description}, {style} style, {background} background, professional lighting, high quality, detailed", "required": True},
                        {"name": "clip", "type": "CLIP", "value": ["1", 1], "required": True}
                    ]
                },
                "3": {
                    "id": "3",
                    "class_type": "CLIPTextEncode",
                    "inputs": [
                        {"name": "text", "type": "STRING", "value": "blurry, low quality, distorted, cluttered background, bad lighting", "required": True},
                        {"name": "clip", "type": "CLIP", "value": ["1", 1], "required": True}
                    ]
                },
                # ... rest of nodes for product generation
            }
        },
        "metadata": {
            "use_case": "e-commerce",
            "product": product_description,
            "style": style,
            "background": background
        }
    }

# Usage example for batch product generation
products = [
    {"name": "wireless headphones", "style": "modern tech", "background": "gradient"},
    {"name": "leather handbag", "style": "luxury fashion", "background": "marble"},
    {"name": "coffee mug", "style": "minimalist", "background": "white"}
]

execution_ids = []
for product in products:
    workflow = create_product_workflow(
        product["name"], 
        product["style"], 
        product["background"]
    )
    execution_id = client.execute_workflow(workflow, priority="high")
    execution_ids.append(execution_id)

# Monitor all executions
for execution_id in execution_ids:
    result = client.wait_for_completion(execution_id)
    if result.status == WorkflowStatus.COMPLETED:
        print(f"Generated product images for {execution_id}")
```

### 2. Social Media Content Creation

Automated content generation for social media platforms.

```python
# Social media content workflow template
def create_social_media_workflow(content_type, theme, platform_specs):
    aspect_ratios = {
        "instagram_post": (1024, 1024),
        "instagram_story": (1080, 1920),
        "facebook_post": (1200, 630),
        "twitter_header": (1500, 500),
        "linkedin_post": (1200, 627)
    }
    
    width, height = aspect_ratios.get(platform_specs["format"], (1024, 1024))
    
    return {
        "workflow": {
            "nodes": {
                "1": {
                    "id": "1",
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": [{"name": "ckpt_name", "type": "STRING", "value": "sd_xl_base_1.0.safetensors", "required": True}]
                },
                "2": {
                    "id": "2",
                    "class_type": "CLIPTextEncode",
                    "inputs": [
                        {"name": "text", "type": "STRING", "value": f"{content_type}, {theme} theme, social media style, eye-catching, vibrant colors, engaging composition, {platform_specs.get('style', 'modern')}", "required": True},
                        {"name": "clip", "type": "CLIP", "value": ["1", 1], "required": True}
                    ]
                },
                "3": {
                    "id": "3",
                    "class_type": "EmptyLatentImage",
                    "inputs": [
                        {"name": "width", "type": "INT", "value": width, "required": True},
                        {"name": "height", "type": "INT", "value": height, "required": True},
                        {"name": "batch_size", "type": "INT", "value": platform_specs.get("variations", 3), "required": True}
                    ]
                },
                # ... rest of workflow nodes
            }
        },
        "metadata": {
            "use_case": "social_media",
            "platform": platform_specs["format"],
            "content_type": content_type,
            "theme": theme,
            "dimensions": f"{width}x{height}"
        }
    }

# Batch social media content generation
content_requests = [
    {
        "content_type": "motivational quote background",
        "theme": "sunrise inspiration",
        "platform": {"format": "instagram_post", "variations": 3, "style": "inspirational"}
    },
    {
        "content_type": "product showcase",
        "theme": "tech innovation",
        "platform": {"format": "linkedin_post", "variations": 2, "style": "professional"}
    }
]

# Schedule content creation
for request in content_requests:
    workflow = create_social_media_workflow(
        request["content_type"],
        request["theme"],
        request["platform"]
    )
    
    execution_id = client.execute_workflow(
        workflow,
        priority="normal",
        metadata={
            "campaign": "weekly_content",
            "schedule_date": "2024-01-15"
        }
    )
    print(f"Scheduled content creation: {execution_id}")
```

### 3. Architectural Visualization

Generate architectural concepts and visualizations.

```python
# Architectural visualization workflow
def create_architecture_workflow(building_type, style, environment):
    return {
        "workflow": {
            "nodes": {
                "1": {
                    "id": "1",
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": [{"name": "ckpt_name", "type": "STRING", "value": "architecture_diffusion_v1.safetensors", "required": True}]
                },
                "2": {
                    "id": "2",
                    "class_type": "CLIPTextEncode",
                    "inputs": [
                        {"name": "text", "type": "STRING", "value": f"architectural visualization, {building_type}, {style} architecture, {environment} setting, professional rendering, detailed, realistic lighting, high quality", "required": True},
                        {"name": "clip", "type": "CLIP", "value": ["1", 1], "required": True}
                    ]
                },
                "3": {
                    "id": "3",
                    "class_type": "CLIPTextEncode",
                    "inputs": [
                        {"name": "text", "type": "STRING", "value": "blurry, low quality, unrealistic proportions, poor lighting", "required": True},
                        {"name": "clip", "type": "CLIP", "value": ["1", 1], "required": True}
                    ]
                },
                "4": {
                    "id": "4",
                    "class_type": "EmptyLatentImage",
                    "inputs": [
                        {"name": "width", "type": "INT", "value": 1024, "required": True},
                        {"name": "height", "type": "INT", "value": 768, "required": True},
                        {"name": "batch_size", "type": "INT", "value": 1, "required": True}
                    ]
                },
                "5": {
                    "id": "5",
                    "class_type": "KSampler",
                    "inputs": [
                        {"name": "seed", "type": "INT", "value": -1, "required": True},
                        {"name": "steps", "type": "INT", "value": 35, "required": True},
                        {"name": "cfg", "type": "FLOAT", "value": 8.5, "required": True},
                        {"name": "sampler_name", "type": "STRING", "value": "dpmpp_2m", "required": True},
                        {"name": "scheduler", "type": "STRING", "value": "karras", "required": True},
                        {"name": "denoise", "type": "FLOAT", "value": 1.0, "required": True},
                        {"name": "model", "type": "MODEL", "value": ["1", 0], "required": True},
                        {"name": "positive", "type": "CONDITIONING", "value": ["2", 0], "required": True},
                        {"name": "negative", "type": "CONDITIONING", "value": ["3", 0], "required": True},
                        {"name": "latent_image", "type": "LATENT", "value": ["4", 0], "required": True}
                    ]
                },
                # Add upscaling for high-resolution architectural renders
                "6": {
                    "id": "6",
                    "class_type": "VAEDecode",
                    "inputs": [
                        {"name": "samples", "type": "LATENT", "value": ["5", 0], "required": True},
                        {"name": "vae", "type": "VAE", "value": ["1", 2], "required": True}
                    ]
                },
                "7": {
                    "id": "7",
                    "class_type": "UpscaleModelLoader",
                    "inputs": [{"name": "model_name", "type": "STRING", "value": "RealESRGAN_x4plus.pth", "required": True}]
                },
                "8": {
                    "id": "8",
                    "class_type": "ImageUpscaleWithModel",
                    "inputs": [
                        {"name": "upscale_model", "type": "UPSCALE_MODEL", "value": ["7", 0], "required": True},
                        {"name": "image", "type": "IMAGE", "value": ["6", 0], "required": True}
                    ]
                },
                "9": {
                    "id": "9",
                    "class_type": "SaveImage",
                    "inputs": [
                        {"name": "images", "type": "IMAGE", "value": ["8", 0], "required": True},
                        {"name": "filename_prefix", "type": "STRING", "value": f"architecture_{building_type.replace(' ', '_')}", "required": False}
                    ]
                }
            }
        },
        "metadata": {
            "use_case": "architectural_visualization",
            "building_type": building_type,
            "style": style,
            "environment": environment,
            "resolution": "high"
        }
    }

# Generate architectural concepts
architectural_projects = [
    {"building": "modern office tower", "style": "contemporary glass", "environment": "urban cityscape"},
    {"building": "residential complex", "style": "sustainable green", "environment": "suburban park"},
    {"building": "cultural center", "style": "avant-garde", "environment": "downtown plaza"}
]

for project in architectural_projects:
    workflow = create_architecture_workflow(
        project["building"],
        project["style"],
        project["environment"]
    )
    
    execution_id = client.execute_workflow(
        workflow,
        priority="high",
        timeout_minutes=45,  # Architecture rendering takes longer
        metadata={"project_type": "concept_design", "client": "architecture_firm"}
    )
    
    print(f"Generated architecture concept: {execution_id}")
```

## Integration Patterns

### 1. Webhook-Based Workflow

Automated workflow triggered by external events.

```python
from flask import Flask, request, jsonify
import hmac
import hashlib

app = Flask(__name__)

@app.route('/webhook/content-request', methods=['POST'])
def handle_content_request():
    """Handle incoming content generation requests via webhook."""
    
    # Verify webhook signature (security best practice)
    signature = request.headers.get('X-Webhook-Signature')
    if not verify_webhook_signature(request.data, signature):
        return jsonify({'error': 'Invalid signature'}), 401
    
    data = request.json
    
    try:
        # Generate workflow based on request type
        workflow_templates = {
            'product_photo': create_product_workflow,
            'social_media': create_social_media_workflow,
            'blog_header': create_blog_header_workflow
        }
        
        workflow_func = workflow_templates.get(data['type'])
        if not workflow_func:
            return jsonify({'error': 'Unknown workflow type'}), 400
        
        # Create workflow
        workflow = workflow_func(
            data.get('description', ''),
            data.get('style', 'modern'),
            data.get('specifications', {})
        )
        
        # Submit to ComfyUI API
        execution_id = client.execute_workflow(
            workflow,
            priority=data.get('priority', 'normal'),
            webhook_url=f"https://your-app.com/webhook/completion/{data['request_id']}",
            metadata={
                'request_id': data['request_id'],
                'user_id': data.get('user_id'),
                'campaign': data.get('campaign')
            }
        )
        
        # Store execution mapping
        store_execution_mapping(data['request_id'], execution_id)
        
        return jsonify({
            'status': 'submitted',
            'execution_id': execution_id,
            'estimated_completion': calculate_eta(workflow)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/webhook/completion/<request_id>', methods=['POST'])
def handle_completion(request_id):
    """Handle workflow completion notifications."""
    
    data = request.json
    execution_id = data['execution_id']
    
    if data['status'] == 'completed':
        # Process completed workflow
        result = client.get_workflow_result(execution_id)
        
        # Upload results to your storage
        image_urls = upload_results_to_storage(result.outputs)
        
        # Notify your application
        notify_application(request_id, {
            'status': 'completed',
            'images': image_urls,
            'execution_time': result.duration_seconds
        })
        
    elif data['status'] == 'failed':
        # Handle failure
        notify_application(request_id, {
            'status': 'failed',
            'error': data.get('error', 'Unknown error')
        })
    
    return jsonify({'status': 'received'})

def verify_webhook_signature(payload, signature):
    """Verify webhook signature for security."""
    secret = os.getenv('WEBHOOK_SECRET').encode()
    expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)

def calculate_eta(workflow):
    """Calculate estimated time for workflow completion."""
    node_count = len(workflow['workflow']['nodes'])
    base_time = 60  # Base 60 seconds
    return base_time + (node_count * 10)  # 10 seconds per node

if __name__ == '__main__':
    app.run(debug=True)
```

### 2. Queue-Based Processing

Using message queues for scalable workflow processing.

```python
import celery
from celery import Celery
import redis
import json

# Celery configuration
celery_app = Celery('comfyui_workflows')
celery_app.config_from_object({
    'broker_url': 'redis://localhost:6379/0',
    'result_backend': 'redis://localhost:6379/0',
    'task_serializer': 'json',
    'accept_content': ['json'],
    'result_serializer': 'json',
    'task_routes': {
        'workflows.generate_image': {'queue': 'image_generation'},
        'workflows.batch_process': {'queue': 'batch_processing'},
    }
})

@celery_app.task(bind=True, max_retries=3)
def generate_image(self, workflow_config, priority='normal'):
    """Celery task for image generation."""
    try:
        client = ComfyUIClient()
        
        # Submit workflow
        execution_id = client.execute_workflow(
            workflow_config['workflow'],
            priority=priority,
            metadata=workflow_config.get('metadata', {})
        )
        
        # Wait for completion
        result = client.wait_for_completion(
            execution_id, 
            timeout_seconds=workflow_config.get('timeout', 1800)
        )
        
        if result.status == WorkflowStatus.COMPLETED:
            return {
                'status': 'completed',
                'execution_id': execution_id,
                'outputs': result.outputs,
                'duration': result.duration_seconds
            }
        else:
            raise Exception(f"Workflow failed: {result.error}")
            
    except Exception as exc:
        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            countdown = 2 ** self.request.retries
            raise self.retry(countdown=countdown, exc=exc)
        else:
            return {
                'status': 'failed',
                'error': str(exc),
                'retries': self.request.retries
            }

@celery_app.task
def batch_process_images(workflow_configs):
    """Process multiple images in batch."""
    results = []
    
    # Submit all workflows
    job_group = celery.group(
        generate_image.s(config, config.get('priority', 'normal'))
        for config in workflow_configs
    )
    
    # Execute batch
    result = job_group.apply_async()
    
    # Wait for all to complete
    for task_result in result.get():
        results.append(task_result)
    
    return {
        'batch_id': result.id,
        'total_jobs': len(workflow_configs),
        'successful': len([r for r in results if r['status'] == 'completed']),
        'failed': len([r for r in results if r['status'] == 'failed']),
        'results': results
    }

# Usage example
def queue_image_generation(prompts, style_configs):
    """Queue multiple image generations."""
    workflow_configs = []
    
    for i, prompt in enumerate(prompts):
        config = style_configs[i % len(style_configs)]
        workflow = create_text_to_image_workflow(prompt, config)
        
        workflow_configs.append({
            'workflow': workflow,
            'metadata': {
                'batch_index': i,
                'prompt': prompt,
                'timestamp': time.time()
            },
            'priority': 'normal',
            'timeout': 1800
        })
    
    # Queue batch processing
    task = batch_process_images.delay(workflow_configs)
    return task.id

# Monitor queue status
@celery_app.task
def get_queue_status():
    """Get current queue status."""
    inspector = celery_app.control.inspect()
    
    return {
        'active': inspector.active(),
        'scheduled': inspector.scheduled(),
        'reserved': inspector.reserved(),
        'stats': inspector.stats()
    }
```

## Performance Optimization

### 1. Workflow Optimization Patterns

Optimize workflows for better performance and resource usage.

```python
class WorkflowOptimizer:
    """Optimize ComfyUI workflows for better performance."""
    
    def __init__(self):
        self.optimization_rules = {
            'reduce_steps': self._optimize_sampler_steps,
            'batch_consolidation': self._consolidate_batch_operations,
            'model_reuse': self._optimize_model_loading,
            'resolution_scaling': self._optimize_resolution
        }
    
    def optimize_workflow(self, workflow, target_performance='balanced'):
        """Apply optimization rules to workflow."""
        optimized = workflow.copy()
        
        performance_profiles = {
            'speed': ['reduce_steps', 'resolution_scaling'],
            'quality': ['model_reuse', 'batch_consolidation'],
            'balanced': ['reduce_steps', 'model_reuse', 'batch_consolidation']
        }
        
        rules_to_apply = performance_profiles.get(target_performance, ['balanced'])
        
        for rule in rules_to_apply:
            if rule in self.optimization_rules:
                optimized = self.optimization_rules[rule](optimized)
        
        return optimized
    
    def _optimize_sampler_steps(self, workflow):
        """Reduce sampling steps for faster generation."""
        nodes = workflow['workflow']['nodes']
        
        for node_id, node in nodes.items():
            if node['class_type'] == 'KSampler':
                for input_param in node['inputs']:
                    if input_param['name'] == 'steps':
                        # Reduce steps while maintaining quality
                        current_steps = input_param['value']
                        optimized_steps = max(15, int(current_steps * 0.75))
                        input_param['value'] = optimized_steps
                        
                        # Also optimize sampler choice
                        if input_param['name'] == 'sampler_name':
                            # Use faster samplers
                            fast_samplers = ['euler', 'euler_a', 'heun']
                            if input_param['value'] not in fast_samplers:
                                input_param['value'] = 'euler_a'
        
        return workflow
    
    def _consolidate_batch_operations(self, workflow):
        """Consolidate operations that can be batched."""
        nodes = workflow['workflow']['nodes']
        
        # Find nodes that can benefit from batching
        for node_id, node in nodes.items():
            if node['class_type'] == 'EmptyLatentImage':
                for input_param in node['inputs']:
                    if input_param['name'] == 'batch_size':
                        # Increase batch size if reasonable
                        if input_param['value'] == 1:
                            input_param['value'] = 2  # Generate 2 variations
        
        return workflow
    
    def _optimize_model_loading(self, workflow):
        """Optimize model loading and reuse."""
        nodes = workflow['workflow']['nodes']
        
        # Track model loaders
        model_loaders = {}
        
        for node_id, node in nodes.items():
            if node['class_type'] in ['CheckpointLoaderSimple', 'LoraLoader']:
                model_name = None
                for input_param in node['inputs']:
                    if input_param['name'] in ['ckpt_name', 'lora_name']:
                        model_name = input_param['value']
                        break
                
                if model_name:
                    if model_name in model_loaders:
                        # Reuse existing loader
                        pass  # Could implement loader sharing
                    else:
                        model_loaders[model_name] = node_id
        
        return workflow
    
    def _optimize_resolution(self, workflow):
        """Optimize image resolution for performance."""
        nodes = workflow['workflow']['nodes']
        
        for node_id, node in nodes.items():
            if node['class_type'] == 'EmptyLatentImage':
                for input_param in node['inputs']:
                    if input_param['name'] in ['width', 'height']:
                        # Reduce resolution for faster processing
                        current_res = input_param['value']
                        if current_res > 768:
                            input_param['value'] = 768
                        elif current_res > 512:
                            input_param['value'] = 512
        
        return workflow

# Usage example
optimizer = WorkflowOptimizer()

# Original workflow
original_workflow = create_text_to_image_workflow(
    "beautiful landscape",
    seed=42,
    steps=30,
    width=1024,
    height=1024
)

# Optimize for speed
speed_optimized = optimizer.optimize_workflow(original_workflow, 'speed')

# Optimize for quality
quality_optimized = optimizer.optimize_workflow(original_workflow, 'quality')

# Submit optimized workflows
execution_ids = [
    client.execute_workflow(speed_optimized, priority='high'),
    client.execute_workflow(quality_optimized, priority='normal')
]
```

### 2. Resource Management Patterns

Implement smart resource management for efficient API usage.

```python
class ResourceManager:
    """Manage ComfyUI API resources efficiently."""
    
    def __init__(self, client):
        self.client = client
        self.model_cache = {}
        self.execution_queue = []
        self.resource_limits = {
            'max_concurrent_executions': 5,
            'memory_threshold_mb': 20000,
            'gpu_threshold_percent': 90
        }
    
    def should_delay_execution(self):
        """Check if execution should be delayed due to resource constraints."""
        try:
            metrics = self.client._make_request('GET', '/metrics/')
            data = metrics.json()
            
            # Check GPU usage
            if data.get('gpu_memory_usage_percent', 0) > self.resource_limits['gpu_threshold_percent']:
                return True, "GPU memory threshold exceeded"
            
            # Check active executions
            if data.get('active_executions', 0) >= self.resource_limits['max_concurrent_executions']:
                return True, "Max concurrent executions reached"
            
            return False, None
            
        except Exception as e:
            return False, f"Could not check resources: {e}"
    
    def smart_execute_workflow(self, workflow, max_retries=3, delay_between_checks=30):
        """Execute workflow with smart resource management."""
        
        for attempt in range(max_retries):
            should_delay, reason = self.should_delay_execution()
            
            if should_delay:
                print(f"Delaying execution: {reason}")
                time.sleep(delay_between_checks)
                continue
            
            # Check if models need to be loaded
            self._ensure_models_loaded(workflow)
            
            try:
                execution_id = self.client.execute_workflow(workflow)
                print(f"Workflow submitted: {execution_id}")
                return execution_id
                
            except Exception as e:
                if "rate limit" in str(e).lower():
                    print(f"Rate limited, waiting...")
                    time.sleep(60)  # Wait 1 minute for rate limit reset
                else:
                    raise
        
        raise Exception(f"Could not execute workflow after {max_retries} attempts")
    
    def _ensure_models_loaded(self, workflow):
        """Ensure required models are loaded before execution."""
        required_models = self._extract_required_models(workflow)
        
        for model_info in required_models:
            if not self._is_model_loaded(model_info):
                print(f"Loading model: {model_info['name']}")
                self.client.load_model(model_info['name'], model_info['type'])
                
                # Wait for model to load
                time.sleep(10)
    
    def _extract_required_models(self, workflow):
        """Extract required models from workflow."""
        models = []
        nodes = workflow['workflow']['nodes']
        
        for node in nodes.values():
            if node['class_type'] == 'CheckpointLoaderSimple':
                for input_param in node['inputs']:
                    if input_param['name'] == 'ckpt_name':
                        models.append({
                            'name': input_param['value'],
                            'type': 'checkpoint'
                        })
            elif node['class_type'] == 'LoraLoader':
                for input_param in node['inputs']:
                    if input_param['name'] == 'lora_name':
                        models.append({
                            'name': input_param['value'],
                            'type': 'lora'
                        })
        
        return models
    
    def _is_model_loaded(self, model_info):
        """Check if model is currently loaded."""
        try:
            status = self.client._make_request('GET', f"/models/{model_info['name']}")
            data = status.json()
            return data.get('is_loaded', False)
        except:
            return False
    
    def cleanup_unused_models(self, max_age_hours=2):
        """Clean up models that haven't been used recently."""
        try:
            result = self.client._make_request('POST', '/models/cleanup', 
                                             params={'max_age_hours': max_age_hours})
            data = result.json()
            print(f"Cleaned up {len(data.get('models_unloaded', []))} unused models")
        except Exception as e:
            print(f"Model cleanup failed: {e}")

# Usage example
resource_manager = ResourceManager(client)

# Smart workflow execution with resource management
workflow = create_text_to_image_workflow("epic fantasy scene")

try:
    execution_id = resource_manager.smart_execute_workflow(workflow)
    result = client.wait_for_completion(execution_id)
    print(f"Workflow completed: {result.status}")
    
    # Clean up resources periodically
    resource_manager.cleanup_unused_models()
    
except Exception as e:
    print(f"Execution failed: {e}")
```

This comprehensive workflow patterns documentation provides real-world examples and production-ready code for various ComfyUI use cases, from basic patterns to advanced integration and optimization strategies.