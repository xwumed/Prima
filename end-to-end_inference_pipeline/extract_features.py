"""
Feature Extraction Pipeline for Prima

This script loads a MRI study (DICOM series), tokenizes it using VQ-VAE,
and passes it through the Prima model to extract CLIP embeddings.

Usage:
    python end-to-end_inference_pipeline/extract_features.py --config configs/pipeline_config.yaml

Output:
    Saves a .pt file containing the embeddings and study ID to the output directory.
"""

import os
import sys
from pathlib import Path

# Ensure repo root is on path so "tools" and other packages import correctly
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import SimpleITK as sitk
import torch
import json
import argparse
import logging
import yaml
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import gc
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from tools.DicomUtils import DicomUtils
from tools.models import ModelLoader
from tools.mrcommondataset import MrVoxelDataset
from tools.utilities import chartovec, convert_serienames_to_tensor, filtercoords
from Prima_training_and_evaluation.patchify import MedicalImagePatchifier


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    study_dir: str
    output_dir: str
    tokenizer_model_config: str
    prima_model_config: str
    batch_size: int = 1
    num_workers: int = 2
    max_tokens_per_chunk: int = 400
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create a PipelineConfig from a dictionary."""
        required_keys = ['study_dir', 'output_dir', 'tokenizer_model_config', 'prima_model_config']
        # We allow missing keys if they have defaults, but check required ones
        missing_keys = [key for key in required_keys if key not in config_dict]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
        
        # Filter out keys that are not in the dataclass
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)


class FeatureExtractor:
    def __init__(self, config: Dict[str, Any]):
        # Extract internal-only keys before passing to PipelineConfig
        batch_log_handler = config.pop('_batch_log_handler', None)
        self.config = PipelineConfig.from_dict(config)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_logging(batch_log_handler)
        self.logger.info(f'Initializing feature extractor with config: {self.config}')
        
        self.tokenizer_model = None
        self.prima_model = None
        self.patchifier = MedicalImagePatchifier(in_dim=256)

    def _setup_logging(self, batch_log_handler=None) -> None:
        log_file = self.output_dir / 'feature_extraction.log'
        # Use a unique logger name per instance so each case gets its own file handler.
        # logging.basicConfig only takes effect once per process, so it cannot be
        # used to redirect logs to different files in batch mode.
        logger_name = f"feature_extractor.{id(self)}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # Don't bubble up to root logger

        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        self.logger.addHandler(sh)

        self._log_handlers = [fh, sh]  # Keep reference for cleanup

        # Shared batch log handler (batch mode only)
        if batch_log_handler is not None:
            self.logger.addHandler(batch_log_handler)
            # Don't close it here; the caller owns it

    def _cleanup(self) -> None:
        self.logger.info("Cleaning up resources...")
        if self.tokenizer_model is not None:
            del self.tokenizer_model
            self.tokenizer_model = None
        if self.prima_model is not None:
            del self.prima_model
            self.prima_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        # Close and remove log handlers to avoid file handle leaks in batch mode
        for handler in getattr(self, '_log_handlers', []):
            handler.close()
            self.logger.removeHandler(handler)




    def _resize_image(
        self,
        image: sitk.Image,
        new_size: Tuple[Optional[int], Optional[int], Optional[int]] = (256, 256, None)
    ) -> sitk.Image:
        """Resize image to new size (axial plane), keeping z spacing."""
        if new_size[0] is None or new_size[1] is None:
            return image
            
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()

        # Calculate new spacing for X and Y
        new_spacing = [
            (original_size[0] * original_spacing[0]) / new_size[0],
            (original_size[1] * original_spacing[1]) / new_size[1],
            original_spacing[2] 
        ]
        
        target_size = list(original_size)
        target_size[0] = new_size[0]
        target_size[1] = new_size[1]
        
        # We don't change Z here unless explicitly requested (which we usually don't for in-plane resize)
        if new_size[2] is not None:
             target_size[2] = new_size[2]
             new_spacing[2] = (original_size[2] * original_spacing[2]) / new_size[2]

        # Create the reference image with new size and spacing
        reference_image = sitk.Image(target_size, image.GetPixelIDValue())
        reference_image.SetOrigin(image.GetOrigin())
        reference_image.SetDirection(image.GetDirection())
        reference_image.SetSpacing(new_spacing)

        # resample
        return sitk.Resample(image, reference_image, sitk.Transform(),
                             sitk.sitkLinear, image.GetPixelIDValue())

    def load_mri_study(self) -> Tuple[List[sitk.Image], List[str]]:
        self.logger.info('Loading MRI study')
        study_path = Path(self.config.study_dir)
        
        # Check for NIfTI files first
        nifti_files = sorted(list(study_path.glob("*.nii")) + list(study_path.glob("*.nii.gz")))
        
        if nifti_files:
            self.logger.info(f"Found {len(nifti_files)} NIfTI files. Loading as NIfTI study.")
            mri_study = []
            series_names = []
            
            for nifti_path in nifti_files:
                try:
                    self.logger.info(f"Processing {nifti_path.name}...")
                    image = sitk.ReadImage(str(nifti_path))
                    
                    # Preprocessing to match DicomUtils.read_dicom_series:
                    # 1. Resize to 256x256 in-plane (first, same as DICOM)
                    image = self._resize_image(image, new_size=(256, 256, None))
                    
                    # 2. Reorient to LPS (second, same as DICOM)
                    image = sitk.DICOMOrient(image, 'LPS')
                    
                    # Extract series name: strip suffix, then take only the last
                    # underscore-separated token to get the modality label.
                    # e.g. "20151002092132_ZHANG_XIUYUN(_D_)_t1c.nii.gz" → "t1c"
                    stem = nifti_path.name.replace(".nii.gz", "").replace(".nii", "")
                    series_name = stem.rsplit("_", 1)[-1]
                    # Clean any remaining special characters
                    series_name = DicomUtils.replace_special_characters(series_name)
                    
                    mri_study.append(image)
                    series_names.append(series_name)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load NIfTI {nifti_path.name}: {e}")
                    continue
            
            if not mri_study:
                raise RuntimeError("No valid NIfTI files could be loaded.")
                
            self.mri_study = mri_study
            self.series_list = series_names
            return self.mri_study, self.series_list

        # Fallback to DICOM loading
        try:
            self.mri_study, self.series_list = DicomUtils.load_mri_study(self.config.study_dir)
            self.logger.info(f'Successfully loaded {len(self.mri_study)} series (DICOM)')
            return self.mri_study, self.series_list
        except Exception as e:
            self.logger.error(f'Failed to load MRI study: {str(e)}')
            raise
    
    def load_tokenizer_model(self) -> torch.nn.Module:
        if self.tokenizer_model is None:
            self.logger.info('Loading tokenizer model')
            try:
                if isinstance(self.config.tokenizer_model_config, str):
                    p = Path(self.config.tokenizer_model_config)
                    with open(p, 'r') as f:
                        tokenizer_config = yaml.safe_load(f) if p.suffix in ('.yaml', '.yml') else json.load(f)
                else:
                    tokenizer_config = self.config.tokenizer_model_config
                self.tokenizer_model = ModelLoader.load_vqvae_model(tokenizer_config)
                self.tokenizer_model = self.tokenizer_model.to(self.config.device)
                self.tokenizer_model.eval()
            except Exception as e:
                self.logger.error(f'Failed to load tokenizer model: {str(e)}')
                raise
        return self.tokenizer_model
    
    def load_full_prima_model(self) -> torch.nn.Module:
        if self.prima_model is None:
            self.logger.info('Loading Prima model')
            try:
                if isinstance(self.config.prima_model_config, str):
                    config_path = Path(self.config.prima_model_config)
                    with open(config_path, 'r') as f:
                        prima_config = yaml.safe_load(f) if config_path.suffix in ('.yaml', '.yml') else json.load(f)
                    
                    # Resolve relative paths
                    config_dir = config_path.resolve().parent
                    if "full_model_ckpt" in prima_config:
                        p = Path(prima_config["full_model_ckpt"])
                        if not p.is_absolute():
                            prima_config = {**prima_config, "full_model_ckpt": str(config_dir / p)}
                else:
                    prima_config = self.config.prima_model_config
                
                self.prima_model = ModelLoader.load_full_prima_model(prima_config)
                self.prima_model = self.prima_model.to(self.config.device)
                self.prima_model.eval()
            except Exception as e:
                self.logger.error(f'Failed to load Prima model: {str(e)}')
                raise
        return self.prima_model
    
    def create_dataset(self, mri_study: List[sitk.Image]) -> DataLoader:
        try:
            dataset = MrVoxelDataset(mri_study)
            num_workers = 0 if 'cuda' in str(self.config.device) else self.config.num_workers
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            return dataloader
        except Exception as e:
            self.logger.error(f'Failed to create dataset: {str(e)}')
            raise

    def run_tokenizer_model(
        self,
        mri_study: List[sitk.Image],
        series_names: Optional[List[str]] = None,
    ) -> Tuple[List[torch.Tensor], List[str], List[Dict]]:
        """Run VQ-VAE tokenizer."""
        self.logger.info('Running tokenizer model')
        vqvae = self.load_tokenizer_model()
        dataloader = self.create_dataset(mri_study)
        series_embeddings = []
        filtered_names = [] if series_names is not None else None
        all_ser_emb_meta = []
        
        try:
            with torch.no_grad():
                for idx, batch in enumerate(tqdm(dataloader, desc="Processing series")):
                    series_name = (series_names[idx] if series_names is not None else f"series_{idx}")
                    try:
                        batch, ser_emb_meta = batch
                        token_list = []
                        tokens = batch[0]
                        num_tokens = tokens.shape[0]

                        num_chunks = (num_tokens + self.config.max_tokens_per_chunk - 1) // self.config.max_tokens_per_chunk
                        for j in range(num_chunks):
                            start_idx = j * self.config.max_tokens_per_chunk
                            end_idx = min((j + 1) * self.config.max_tokens_per_chunk, num_tokens)
                            chunk = tokens[start_idx:end_idx].unsqueeze(1)
                            token_list.append(chunk)

                        embeddings = [
                            vqvae.encode(chunk.to(self.config.device)).detach().cpu()
                            for chunk in token_list
                        ]
                        series_embedding = torch.cat(embeddings, dim=0)
                        series_embeddings.append(series_embedding)
                        if filtered_names is not None:
                            filtered_names.append(series_name)
                        all_ser_emb_meta.append(ser_emb_meta)
                    except Exception as e:
                        self.logger.warning(
                            f"Skipping series index={idx} name={series_name} due to error: {str(e)}"
                        )
                        continue

            return series_embeddings, (filtered_names if series_names is not None else None), all_ser_emb_meta
        finally:
            # NOTE: In single-study mode, release tokenizer to free GPU memory before loading Prima.
            # In batch mode (BatchExtractor), the caller sets keep_models=True to skip this.
            if not getattr(self, '_keep_models', False):
                self.tokenizer_model = None
                torch.cuda.empty_cache()
                gc.collect()

    def prepare_prima_input(
        self,
        series_embeddings: List[torch.Tensor],
        series_names: List[str],
        all_ser_emb_meta: Optional[List[Dict[str, Any]]] = None,
        otsu_percentage: int = 5,
    ) -> Dict[str, Any]:
        """Prepare input for Prima model, with OTSU background filtering (matches pipeline.py)."""
        coords = None

        if all_ser_emb_meta is not None:
            # OTSU filtering: remove background tokens, keep at least 25 foreground tokens.
            # Try decreasing OTSU threshold until enough tokens remain.
            coords = []
            new_series_embeddings = []
            for i, ser_emb_meta in enumerate(all_ser_emb_meta):
                for percent in range(otsu_percentage, -1, -1):
                    embs, embspos, _ = filtercoords(ser_emb_meta, percent, series_embeddings[i])
                    self.logger.info(
                        f"OTSU {percent}% for '{series_names[i]}': "
                        f"{len(series_embeddings[i])} → {len(embs)} tokens"
                    )
                    if len(embspos) > 25:
                        break
                new_series_embeddings.append(embs)
                coords.append(embspos)
            series_embeddings = new_series_embeddings

        study_lens = torch.tensor([len(series_embeddings)], dtype=torch.long)
        serie_lenss = torch.tensor([len(v) for v in series_embeddings], dtype=torch.long).unsqueeze(0)

        # Patchify (with OTSU-filtered coordinates if available)
        patched = self.patchifier(series_embeddings, coords=coords)
        max_len = serie_lenss.max()
        visuals = []
        for img in patched:
            sizes = list(img.shape)
            img_pad_len = max_len - len(img)
            sizes[0] = img_pad_len
            img_pad = torch.zeros(sizes)
            visuals.append(torch.cat([img, img_pad], dim=0).unsqueeze(0))

        # Series name tensors
        seriename_tensors = [chartovec(name) for name in series_names]
        max_seriename_len = max(len(t) for t in seriename_tensors)
        num_series = len(seriename_tensors)
        serienames_tensor = torch.zeros(num_series, max_seriename_len, dtype=torch.long)
        for i, t in enumerate(seriename_tensors):
            serienames_tensor[i, :len(t)] = t
        serienames = serienames_tensor.unsqueeze(0)

        study_desc = chartovec("MR BRAIN W CONTRAST").unsqueeze(0)

        return {
            'visual': visuals,
            'lens': study_lens,
            'lenss': serie_lenss,
            'hash': ["study_0"],
            'serienames': serienames,
            'studydescription': study_desc
        }

    def run_prima_model(
        self,
        series_embeddings: List[torch.Tensor],
        series_names: List[str],
        all_ser_emb_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run Prima model and return full predictions dict (clip_emb, diagnosis, referral, priority)."""
        self.logger.info('Running Prima model to extract features')
        
        def move_to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, list):
                return [move_to_device(item, device) for item in obj]
            elif isinstance(obj, dict):
                return {k: move_to_device(v, device) for k, v in obj.items()}
            return obj

        try:
            # NOTE: In single-study mode, release tokenizer to free GPU memory before loading Prima.
            # In batch mode (BatchExtractor), the caller sets keep_models=True to skip this.
            if not getattr(self, '_keep_models', False):
                if self.tokenizer_model is not None:
                    del self.tokenizer_model
                    self.tokenizer_model = None
                if 'cuda' in str(self.config.device):
                    torch.cuda.empty_cache()
                gc.collect()

            prima_input = self.prepare_prima_input(
                series_embeddings=series_embeddings,
                series_names=series_names,
                all_ser_emb_meta=all_ser_emb_meta,
            )
            prima_input = move_to_device(prima_input, self.config.device)

            if self.prima_model is None:
                self.prima_model = self.load_full_prima_model()
            if hasattr(self.prima_model, 'make_no_flashattn'):
                self.prima_model.make_no_flashattn()

            device_type = 'cuda' if 'cuda' in str(self.config.device) else 'cpu'
            with torch.no_grad():
                if device_type == 'cuda':
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        predictions = self.prima_model(prima_input, inference_only_once=True)
                else:
                    predictions = self.prima_model(prima_input)

            # Free input from GPU before serialization (matches pipeline.py)
            del prima_input
            if 'cuda' in str(self.config.device):
                torch.cuda.empty_cache()

            clip_emb = predictions.get("clip_emb")
            if clip_emb is None:
                raise ValueError("Model output does not contain 'clip_emb'")

            # Move all tensors to CPU and detach
            def to_cpu(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.detach().cpu()
                elif isinstance(obj, dict):
                    return {k: to_cpu(v) for k, v in obj.items()}
                return obj

            return to_cpu(predictions)

        finally:
            # NOTE: In single-study mode, cleanup after each case.
            # In batch mode (BatchExtractor), skip cleanup so models stay loaded.
            if not getattr(self, '_keep_models', False):
                self._cleanup()


def tensor_to_python(obj):
    """Recursively convert tensors to Python scalars/lists for JSON serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_python(v) for k, v in obj.items()}
    return obj


def save_study_outputs(predictions: dict, config: dict, series_names_used: list) -> None:
    """Save .pt features and predictions.json for a single study."""
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    study_id = Path(config['study_dir']).name or "study"

    clip_emb = predictions.get("clip_emb")
    features_path = output_dir / f"{study_id}_features.pt"
    torch.save({
        'study_id': study_id,
        'features': clip_emb,
        'series_names': series_names_used
    }, features_path)
    print(f"  [OK] Features saved to {features_path}  shape={clip_emb.shape}")

    predictions_json = {
        "diagnosis": tensor_to_python(predictions.get("diagnosis", {})),
        "referral": tensor_to_python(predictions.get("referral", {})),
        "priority": tensor_to_python(predictions.get("priority", {})),
    }
    json_path = output_dir / f"{study_id}_predictions.json"
    with open(json_path, 'w') as f:
        json.dump(predictions_json, f, indent=2)
    print(f"  [OK] Predictions saved to {json_path}")


def process_one_study(config: dict) -> None:
    """Process a single study (single-study mode): load models, infer, save, cleanup."""
    extractor = FeatureExtractor(config)

    mri_study, series_names = extractor.load_mri_study()
    series_embeddings, series_names_used, all_ser_emb_meta = extractor.run_tokenizer_model(
        mri_study, series_names=series_names
    )
    if not series_embeddings:
        raise RuntimeError("No series tokenized.")

    predictions = extractor.run_prima_model(
        series_embeddings=series_embeddings,
        series_names=series_names_used,
        all_ser_emb_meta=all_ser_emb_meta,
    )
    save_study_outputs(predictions, config, series_names_used)


class BatchExtractor:
    """
    Batch-mode extractor that loads tokenizer and Prima models ONCE and reuses
    them across all cases, avoiding repeated model loading overhead.

    Usage:
        with BatchExtractor(base_config) as bex:
            for case_dir in case_dirs:
                bex.process(case_dir, output_dir)
    """

    def __init__(self, base_config: dict, batch_log_handler=None):
        self.base_config = base_config
        self.batch_log_handler = batch_log_handler
        self.device = base_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self._tokenizer = None
        self._prima = None
        self._patchifier = MedicalImagePatchifier(in_dim=256)
        self._logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(f"batch_extractor.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if self.batch_log_handler is not None:
            logger.addHandler(self.batch_log_handler)
        return logger

    def _load_models(self):
        """Load tokenizer and Prima model once."""
        self._logger.info("Loading tokenizer model (batch mode, load once)...")
        # Reuse FeatureExtractor's load logic via a dummy extractor
        dummy_config = {
            **self.base_config,
            'study_dir': self.base_config.get('study_dir', '/tmp'),
            'output_dir': self.base_config.get('output_dir', '/tmp'),
        }
        # We need a FeatureExtractor instance just to load models
        self._model_holder = FeatureExtractor.__new__(FeatureExtractor)
        self._model_holder.config = PipelineConfig.from_dict(
            {k: v for k, v in dummy_config.items() if k in PipelineConfig.__dataclass_fields__}
        )
        self._model_holder.output_dir = Path(dummy_config['output_dir'])
        self._model_holder.tokenizer_model = None
        self._model_holder.prima_model = None
        self._model_holder.patchifier = self._patchifier
        self._model_holder._keep_models = True
        self._model_holder._log_handlers = []
        self._model_holder.logger = self._logger

        self._model_holder.load_tokenizer_model()
        self._model_holder.load_full_prima_model()
        if hasattr(self._model_holder.prima_model, 'make_no_flashattn'):
            self._model_holder.prima_model.make_no_flashattn()

        self._logger.info("Both models loaded. Starting batch processing.")

    def _release_models(self):
        """Release all models and free GPU memory."""
        self._logger.info("Releasing models after batch completion.")
        if hasattr(self, '_model_holder'):
            self._model_holder._keep_models = False
            self._model_holder._cleanup()

    def __enter__(self):
        self._load_models()
        return self

    def __exit__(self, *args):
        self._release_models()

    def process(self, case_dir: Path, output_dir: Path, case_log_handler=None) -> None:
        """Process a single case, reusing already-loaded models."""
        case_config = {
            **self.base_config,
            'study_dir': str(case_dir),
            'output_dir': str(output_dir),
        }
        # Build a FeatureExtractor that shares the loaded models
        extractor = FeatureExtractor.__new__(FeatureExtractor)
        extractor.config = PipelineConfig.from_dict(
            {k: v for k, v in case_config.items() if k in PipelineConfig.__dataclass_fields__}
        )
        extractor.output_dir = output_dir
        extractor.output_dir.mkdir(parents=True, exist_ok=True)
        extractor.patchifier = self._patchifier
        extractor._keep_models = True  # Don't cleanup after each case

        # Share the loaded models
        extractor.tokenizer_model = self._model_holder.tokenizer_model
        extractor.prima_model = self._model_holder.prima_model

        # Set up per-case logger
        logger_name = f"feature_extractor.{id(extractor)}"
        extractor.logger = logging.getLogger(logger_name)
        extractor.logger.setLevel(logging.INFO)
        extractor.logger.propagate = False
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_file = output_dir / 'feature_extraction.log'
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        extractor.logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        extractor.logger.addHandler(sh)
        if self.batch_log_handler is not None:
            extractor.logger.addHandler(self.batch_log_handler)
        extractor._log_handlers = [fh, sh]

        try:
            mri_study, series_names = extractor.load_mri_study()
            series_embeddings, series_names_used, all_ser_emb_meta = extractor.run_tokenizer_model(
                mri_study, series_names=series_names
            )
            if not series_embeddings:
                raise RuntimeError("No series tokenized.")

            predictions = extractor.run_prima_model(
                series_embeddings=series_embeddings,
                series_names=series_names_used,
                all_ser_emb_meta=all_ser_emb_meta,
            )
            save_study_outputs(predictions, case_config, series_names_used)
        finally:
            # Close per-case file handler (don't close shared batch handler)
            for h in [fh, sh]:
                h.close()
                extractor.logger.removeHandler(h)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prima Feature Extraction')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument(
        '--study_root', type=str, default=None,
        help=(
            'Optional: root directory containing multiple case subdirectories. '
            'When set, processes each subdirectory as a separate case. '
            'Outputs are saved to <study_root>_prima/<case_id>/. '
            'Example: --study_root /home/xuewei/MRI/test'
        )
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f) if config_path.suffix in ('.yaml', '.yml') else json.load(f)

    if args.study_root:
        # ── Batch mode: models loaded ONCE, reused across all cases ─────────
        study_root = Path(args.study_root)
        case_dirs = sorted([d for d in study_root.iterdir() if d.is_dir()])

        if not case_dirs:
            logging.error(f"No subdirectories found in {study_root}")
            sys.exit(1)

        base_output_dir = study_root.parent / (study_root.name + "_prima")
        base_output_dir.mkdir(parents=True, exist_ok=True)

        batch_log_path = base_output_dir / "batch_extraction.log"
        batch_log_handler = logging.FileHandler(batch_log_path)
        batch_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        print(f"Batch mode: found {len(case_dirs)} cases in {study_root}")
        print(f"Output root: {base_output_dir}")
        print(f"Models will be loaded ONCE and reused across all cases.")

        success, skipped, failed = 0, 0, []
        with BatchExtractor(base_config, batch_log_handler=batch_log_handler) as bex:
            for idx, case_dir in enumerate(case_dirs):
                # ── Skip logic: check if features.pt already exists ──
                out_dir = base_output_dir / case_dir.name
                existing_feats = list(out_dir.glob("*_features.pt")) if out_dir.exists() else []
                if existing_feats:
                    print(f"[{idx+1}/{len(case_dirs)}] {case_dir.name}: already done, skipping")
                    skipped += 1
                    success += 1
                    continue

                print(f"\n{'='*60}")
                print(f"[{idx+1}/{len(case_dirs)}] Processing: {case_dir.name}")
                try:
                    bex.process(case_dir, out_dir)
                    success += 1
                except Exception as e:
                    logging.error(f"  [FAIL] {case_dir.name}: {e}")
                    failed.append(case_dir.name)

        batch_log_handler.close()
        print(f"\n{'='*60}")
        print(f"Batch complete: {success}/{len(case_dirs)} succeeded ({skipped} skipped).")
        print(f"Full batch log: {batch_log_path}")
        if failed:
            print(f"Failed cases: {failed}")

    else:
        # ── Single study mode ────────────────────────────────────────────────
        try:
            process_one_study(base_config)
        except Exception as e:
            logging.error(f"Extraction failed: {str(e)}")
            raise




