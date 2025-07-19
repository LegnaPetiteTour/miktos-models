"""
Miktos AI Models - Core Model Management System
Handles downloading, organizing, and optimizing AI models
"""

import os
import json
import yaml
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from huggingface_hub import hf_hub_download, list_repo_files
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about an AI model"""
    id: str
    name: str
    description: str
    size: str
    category: str
    source: str  # 'huggingface', 'civitai', 'local'
    url: str
    filename: str
    sha256: Optional[str] = None
    requirements: Optional[Dict[str, Any]] = None
    tags: List[str] = None
    version: str = "1.0.0"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.requirements is None:
            self.requirements = {}


class ModelRegistry:
    """Manages the registry of available AI models"""
    
    def __init__(self, registry_path: str = "registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.models: Dict[str, ModelInfo] = {}
        self._load_registries()
    
    def _load_registries(self):
        """Load all registry files"""
        registry_files = [
            self.registry_path / "official.yaml",
            self.registry_path / "community.yaml", 
            self.registry_path / "custom.yaml"
        ]
        
        for registry_file in registry_files:
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    data = yaml.safe_load(f) or {}
                    for model_data in data.get('models', []):
                        model = ModelInfo(**model_data)
                        self.models[model.id] = model
    
    def list_models(self, category: Optional[str] = None) -> List[ModelInfo]:
        """List all available models, optionally filtered by category"""
        models = list(self.models.values())
        if category:
            models = [m for m in models if m.category == category]
        return sorted(models, key=lambda m: m.name)
    
    def search(self, query: str = "", category: str = "", tags: List[str] = None) -> List[ModelInfo]:
        """Search models by query, category, or tags"""
        results = []
        tags = tags or []
        
        for model in self.models.values():
            # Check category filter
            if category and model.category != category:
                continue
                
            # Check tags filter
            if tags and not any(tag in model.tags for tag in tags):
                continue
                
            # Check query filter
            if query:
                query_lower = query.lower()
                if (query_lower in model.name.lower() or 
                    query_lower in model.description.lower() or
                    any(query_lower in tag.lower() for tag in model.tags)):
                    results.append(model)
            else:
                results.append(model)
        
        return sorted(results, key=lambda m: m.name)
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model info by ID"""
        return self.models.get(model_id)
    
    def add_model(self, model_info: ModelInfo):
        """Add a new model to the registry"""
        self.models[model_info.id] = model_info
        
    def save_custom_registry(self):
        """Save custom models to registry file"""
        custom_models = [
            asdict(model) for model in self.models.values() 
            if model.source == 'local'
        ]
        
        custom_file = self.registry_path / "custom.yaml"
        with open(custom_file, 'w') as f:
            yaml.dump({'models': custom_models}, f, default_flow_style=False)


class ModelManager:
    """Manages downloading and organizing AI models"""
    
    def __init__(self, models_path: str = "models"):
        self.models_path = Path(models_path)
        self.models_path.mkdir(exist_ok=True)
        self.registry = ModelRegistry()
        
        # Create category directories
        for category in ['stable-diffusion', 'controlnet', 'lora', 'embeddings', '3d']:
            (self.models_path / category).mkdir(exist_ok=True)
    
    def download(self, model_id: str, progress_callback: Optional[Callable] = None) -> bool:
        """Download a model by ID"""
        model_info = self.registry.get_model(model_id)
        if not model_info:
            logger.error(f"Model {model_id} not found in registry")
            return False
        
        # Determine local path
        category_path = self.models_path / model_info.category
        local_path = category_path / model_info.filename
        
        # Check if already downloaded
        if local_path.exists() and self._verify_file(local_path, model_info.sha256):
            logger.info(f"Model {model_id} already downloaded and verified")
            return True
        
        logger.info(f"Downloading {model_info.name}...")
        
        try:
            if model_info.source == 'huggingface':
                self._download_from_huggingface(model_info, local_path, progress_callback)
            elif model_info.source == 'civitai':
                self._download_from_url(model_info.url, local_path, progress_callback)
            else:
                logger.error(f"Unsupported source: {model_info.source}")
                return False
            
            # Verify download
            if model_info.sha256 and not self._verify_file(local_path, model_info.sha256):
                logger.error(f"Download verification failed for {model_id}")
                local_path.unlink()  # Remove corrupted file
                return False
            
            logger.info(f"Successfully downloaded {model_info.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {model_id}: {e}")
            if local_path.exists():
                local_path.unlink()
            return False
    
    def download_batch(self, model_ids: List[str], progress_callback: Optional[Callable] = None) -> Dict[str, bool]:
        """Download multiple models"""
        results = {}
        for model_id in model_ids:
            results[model_id] = self.download(model_id, progress_callback)
        return results
    
    def _download_from_huggingface(self, model_info: ModelInfo, local_path: Path, progress_callback: Optional[Callable]):
        """Download from Hugging Face Hub"""
        # Extract repo_id from URL
        url_parts = model_info.url.replace('https://huggingface.co/', '').split('/')
        repo_id = '/'.join(url_parts[:2])
        
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_info.filename,
            cache_dir=str(local_path.parent),
            force_download=True
        )
        
        # Move to desired location if needed
        if Path(downloaded_path) != local_path:
            import shutil
            shutil.move(downloaded_path, local_path)
    
    def _download_from_url(self, url: str, local_path: Path, progress_callback: Optional[Callable]):
        """Download from direct URL"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if progress_callback and total_size > 0:
                        progress_callback(downloaded, total_size)
    
    def _verify_file(self, file_path: Path, expected_sha256: str) -> bool:
        """Verify file integrity using SHA256"""
        if not expected_sha256:
            return True
            
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest() == expected_sha256
    
    def list_downloaded(self) -> List[Dict[str, Any]]:
        """List all downloaded models"""
        downloaded = []
        
        for category_dir in self.models_path.iterdir():
            if category_dir.is_dir():
                for model_file in category_dir.iterdir():
                    if model_file.is_file():
                        # Try to find matching model info
                        model_info = None
                        for model in self.registry.models.values():
                            if model.filename == model_file.name:
                                model_info = model
                                break
                        
                        downloaded.append({
                            'path': str(model_file),
                            'size': model_file.stat().st_size,
                            'category': category_dir.name,
                            'model_info': model_info
                        })
        
        return downloaded
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a downloaded model"""
        model_info = self.registry.get_model(model_id)
        if not model_info:
            return False
        
        local_path = self.models_path / model_info.category / model_info.filename
        if local_path.exists():
            local_path.unlink()
            logger.info(f"Deleted {model_info.name}")
            return True
        
        return False


class LoRAManager:
    """Specialized manager for LoRA models"""
    
    def __init__(self, models_path: str = "models"):
        self.lora_path = Path(models_path) / "lora"
        self.lora_path.mkdir(parents=True, exist_ok=True)
        self.presets_path = self.lora_path / "presets"
        self.presets_path.mkdir(exist_ok=True)
    
    def install(self, path: str, category: str = "style", tags: List[str] = None) -> str:
        """Install a LoRA from local path"""
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f"LoRA file not found: {path}")
        
        # Create category subdirectory
        category_path = self.lora_path / category
        category_path.mkdir(exist_ok=True)
        
        # Copy to LoRA directory
        dest_path = category_path / source_path.name
        import shutil
        shutil.copy2(source_path, dest_path)
        
        # Create metadata file
        metadata = {
            'name': source_path.stem,
            'category': category,
            'tags': tags or [],
            'filename': source_path.name,
            'installed_date': str(Path().stat().st_mtime)
        }
        
        metadata_path = dest_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(dest_path)
    
    def search(self, tags: List[str] = None, category: str = None) -> List[Dict[str, Any]]:
        """Search LoRA models by tags or category"""
        results = []
        
        for lora_file in self.lora_path.rglob("*.safetensors"):
            metadata_file = lora_file.with_suffix('.json')
            
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            # Apply filters
            if category and metadata.get('category') != category:
                continue
                
            if tags and not any(tag in metadata.get('tags', []) for tag in tags):
                continue
            
            results.append({
                'path': str(lora_file),
                'name': metadata.get('name', lora_file.stem),
                'category': metadata.get('category', 'unknown'),
                'tags': metadata.get('tags', []),
                'metadata': metadata
            })
        
        return results
    
    def create_preset(self, name: str, loras: List[tuple]) -> str:
        """Create a LoRA preset with multiple LoRAs and strengths"""
        preset_data = {
            'name': name,
            'loras': [
                {'name': lora_name, 'strength': strength}
                for lora_name, strength in loras
            ]
        }
        
        preset_file = self.presets_path / f"{name}.json"
        with open(preset_file, 'w') as f:
            json.dump(preset_data, f, indent=2)
        
        return str(preset_file)
    
    def load_preset(self, name: str) -> Dict[str, Any]:
        """Load a LoRA preset"""
        preset_file = self.presets_path / f"{name}.json"
        if not preset_file.exists():
            raise FileNotFoundError(f"Preset not found: {name}")
        
        with open(preset_file, 'r') as f:
            return json.load(f)


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Miktos AI Model Manager")
    parser.add_argument("--download", help="Download a model by ID")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--search", help="Search models")
    parser.add_argument("--preset", help="Download model preset (essential/professional/complete)")
    
    args = parser.parse_args()
    
    manager = ModelManager()
    
    if args.list:
        models = manager.registry.list_models()
        for model in models:
            print(f"{model.id}: {model.name} ({model.size})")
    
    elif args.search:
        models = manager.registry.search(args.search)
        for model in models:
            print(f"{model.id}: {model.name} - {model.description}")
    
    elif args.download:
        success = manager.download(args.download)
        if success:
            print(f"Successfully downloaded {args.download}")
        else:
            print(f"Failed to download {args.download}")
    
    elif args.preset:
        # Load preset configuration and download models
        preset_models = {
            'essential': ['stable-diffusion-xl-base', 'controlnet-depth'],
            'professional': ['stable-diffusion-xl-base', 'flux-dev', 'controlnet-depth', 'controlnet-canny'],
            'complete': ['stable-diffusion-xl-base', 'flux-dev', 'controlnet-depth', 'controlnet-canny', 'ip-adapter']
        }
        
        if args.preset in preset_models:
            models = preset_models[args.preset]
            results = manager.download_batch(models)
            
            for model_id, success in results.items():
                status = "✅" if success else "❌"
                print(f"{status} {model_id}")
        else:
            print(f"Unknown preset: {args.preset}")
