"""GPU utility functions."""
from typing import Dict, Optional, Any
import structlog

logger = structlog.get_logger()


def get_gpu_memory_info() -> Optional[Dict[str, Any]]:
    """Get GPU memory information."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return None
        
        device = torch.cuda.current_device()
        
        # Get memory info in bytes, convert to MB
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device)
        
        total_mb = total_memory / (1024 ** 2)
        allocated_mb = allocated_memory / (1024 ** 2)
        cached_mb = cached_memory / (1024 ** 2)
        free_mb = total_mb - allocated_mb
        
        return {
            'device_id': device,
            'device_name': torch.cuda.get_device_name(device),
            'total_mb': round(total_mb, 2),
            'allocated_mb': round(allocated_mb, 2),
            'cached_mb': round(cached_mb, 2),
            'free_mb': round(free_mb, 2),
            'used_mb': round(allocated_mb, 2),
            'utilization_percent': round((allocated_mb / total_mb) * 100, 2)
        }
        
    except ImportError:
        logger.debug("PyTorch not available, cannot get GPU memory info")
        return None
    except Exception as e:
        logger.error("Error getting GPU memory info", error=str(e))
        return None


def get_gpu_utilization() -> Optional[Dict[str, Any]]:
    """Get GPU utilization information."""
    try:
        import pynvml
        
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        if device_count == 0:
            return None
        
        # Get info for first GPU (could be extended for multi-GPU)
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # GPU utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = util.gpu
        memory_util = util.memory
        
        # Temperature
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except:
            temp = None
        
        # Power usage
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
        except:
            power = None
        
        # Clock speeds
        try:
            graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        except:
            graphics_clock = None
            memory_clock = None
        
        return {
            'utilization': gpu_util,  # GPU utilization percentage
            'memory_utilization': memory_util,  # Memory utilization percentage
            'temperature': temp,  # Temperature in Celsius
            'power_usage_watts': power,  # Power usage in watts
            'graphics_clock_mhz': graphics_clock,  # Graphics clock in MHz
            'memory_clock_mhz': memory_clock,  # Memory clock in MHz
        }
        
    except ImportError:
        logger.debug("pynvml not available, cannot get GPU utilization")
        
        # Fallback to nvidia-smi if available
        try:
            import subprocess
            import xml.etree.ElementTree as ET
            
            result = subprocess.run([
                'nvidia-smi', '-q', '-x'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                root = ET.fromstring(result.stdout)
                gpu = root.find('gpu')
                
                if gpu is not None:
                    utilization = gpu.find('utilization')
                    temperature = gpu.find('temperature')
                    
                    gpu_util = None
                    temp = None
                    
                    if utilization is not None:
                        gpu_util_elem = utilization.find('gpu_util')
                        if gpu_util_elem is not None:
                            gpu_util = int(gpu_util_elem.text.replace('%', ''))
                    
                    if temperature is not None:
                        gpu_temp_elem = temperature.find('gpu_temp')
                        if gpu_temp_elem is not None:
                            temp = int(gpu_temp_elem.text.replace(' C', ''))
                    
                    return {
                        'utilization': gpu_util,
                        'temperature': temp,
                        'memory_utilization': None,
                        'power_usage_watts': None,
                        'graphics_clock_mhz': None,
                        'memory_clock_mhz': None
                    }
            
        except Exception:
            pass
        
        return None
        
    except Exception as e:
        logger.error("Error getting GPU utilization", error=str(e))
        return None


def check_gpu_requirements(min_memory_gb: float = 4.0) -> Dict[str, Any]:
    """Check if GPU meets minimum requirements."""
    try:
        import torch
        
        result = {
            'cuda_available': torch.cuda.is_available(),
            'meets_requirements': False,
            'gpu_count': 0,
            'total_memory_gb': 0,
            'devices': []
        }
        
        if not torch.cuda.is_available():
            result['message'] = "CUDA not available"
            return result
        
        gpu_count = torch.cuda.device_count()
        result['gpu_count'] = gpu_count
        
        total_memory = 0
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024 ** 3)
            total_memory += memory_gb
            
            device_info = {
                'device_id': i,
                'name': props.name,
                'memory_gb': round(memory_gb, 2),
                'compute_capability': f"{props.major}.{props.minor}",
                'multi_processor_count': props.multi_processor_count
            }
            
            result['devices'].append(device_info)
        
        result['total_memory_gb'] = round(total_memory, 2)
        result['meets_requirements'] = total_memory >= min_memory_gb
        
        if result['meets_requirements']:
            result['message'] = f"GPU requirements met ({total_memory:.1f}GB >= {min_memory_gb}GB)"
        else:
            result['message'] = f"Insufficient GPU memory ({total_memory:.1f}GB < {min_memory_gb}GB)"
        
        return result
        
    except ImportError:
        return {
            'cuda_available': False,
            'meets_requirements': False,
            'message': "PyTorch not available",
            'gpu_count': 0,
            'total_memory_gb': 0,
            'devices': []
        }
    except Exception as e:
        logger.error("Error checking GPU requirements", error=str(e))
        return {
            'cuda_available': False,
            'meets_requirements': False,
            'message': f"Error: {str(e)}",
            'gpu_count': 0,
            'total_memory_gb': 0,
            'devices': []
        }


def optimize_gpu_memory():
    """Optimize GPU memory usage."""
    try:
        import torch
        
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            
            # Set memory fraction if configured
            from src.config.settings import settings
            if hasattr(settings, 'GPU_MEMORY_FRACTION') and settings.GPU_MEMORY_FRACTION < 1.0:
                torch.cuda.set_per_process_memory_fraction(settings.GPU_MEMORY_FRACTION)
            
            logger.info("GPU memory optimized")
            return True
            
    except Exception as e:
        logger.error("Error optimizing GPU memory", error=str(e))
        return False
    
    return False


def get_optimal_batch_size(base_batch_size: int = 1) -> int:
    """Get optimal batch size based on available GPU memory."""
    try:
        gpu_info = get_gpu_memory_info()
        if not gpu_info:
            return base_batch_size
        
        # Simple heuristic: 1 batch per 2GB of free memory
        free_gb = gpu_info['free_mb'] / 1024
        optimal_batch_size = max(1, int(free_gb // 2))
        
        # Limit to reasonable range
        return min(optimal_batch_size, base_batch_size * 4)
        
    except Exception as e:
        logger.error("Error calculating optimal batch size", error=str(e))
        return base_batch_size


def monitor_gpu_memory() -> Dict[str, Any]:
    """Monitor GPU memory and return detailed stats."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}
        
        stats = {}
        device_count = torch.cuda.device_count()
        
        for i in range(device_count):
            device_stats = {
                'device_name': torch.cuda.get_device_name(i),
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_reserved': torch.cuda.memory_reserved(i),
                'max_memory_allocated': torch.cuda.max_memory_allocated(i),
                'max_memory_reserved': torch.cuda.max_memory_reserved(i),
                'memory_summary': torch.cuda.memory_summary(i)
            }
            
            stats[f'cuda:{i}'] = device_stats
        
        return stats
        
    except Exception as e:
        logger.error("Error monitoring GPU memory", error=str(e))
        return {'error': str(e)}