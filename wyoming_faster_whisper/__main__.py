#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import platform
from functools import partial

from huggingface_hub import snapshot_download
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import WhisperTurboEventHandler
from .whisper_turbo import Transcriber

_LOGGER = logging.getLogger(__name__)

async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Name of whisper model to use (or auto)",
    )
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--data-dir",
        required=True,
        action="append",
        help="Data directory to check for downloaded models",
    )
    parser.add_argument(
        "--download-dir",
        help="Directory to download models into (default: first data dir)",
    )
    parser.add_argument(
        "--language",
        help="Default language to set for transcription",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick transcription mode",
    )
    parser.add_argument(
        "--initial-prompt",
        help="Optional text to provide as a prompt for the first window",
    )
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    if not args.download_dir:
        # Download to first data dir by default
        args.download_dir = args.data_dir[0]

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    # Automatic configuration for ARM
    machine = platform.machine().lower()
    is_arm = ("arm" in machine) or ("aarch" in machine)
    if args.model == "auto":
        args.model = "tiny" if is_arm else "base"
        _LOGGER.debug("Model automatically selected: %s", args.model)

    # Load model configuration
    path_hf = snapshot_download(
        repo_id='openai/whisper-large-v3-turbo',
        allow_patterns=["config.json", "model.safetensors"],
        cache_dir=args.download_dir
    )
    
    with open(f'{path_hf}/config.json', 'r') as fp:
        cfg = json.load(fp)

    # Initialize model
    _LOGGER.debug("Loading Whisper Turbo MLX model")
    model = Transcriber(cfg)
    weights = [(k.replace("embed_positions.weight", "positional_embedding"), 
                v.swapaxes(1, 2) if ('conv' in k and v.ndim==3) else v) 
              for k, v in mx.load(f'{path_hf}/model.safetensors').items()]
    model.load_weights(weights, strict=False)
    model.eval()

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="whisper-turbo-mlx",
                description="Whisper Turbo implementation using MLX",
                attribution=Attribution(
                    name="Josef Albers",
                    url="https://github.com/JosefAlbers/whisper-turbo-mlx",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=args.model,
                        description=f"Whisper Turbo {args.model} model",
                        attribution=Attribution(
                            name="OpenAI",
                            url="https://github.com/openai/whisper",
                        ),
                        installed=True,
                        languages=["auto", "en"],  # Update with full language list
                        version="3.0",
                    )
                ],
            )
        ],
    )

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")
    model_lock = asyncio.Lock()
    await server.run(
        partial(
            WhisperTurboEventHandler,
            wyoming_info,
            args,
            model,
            model_lock,
            initial_prompt=args.initial_prompt,
        )
    )

def run() -> None:
    asyncio.run(main())

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
