#!/usr/bin/env python3
"""
Auto-Subtitles: Video/Audio to Subtitle Generator

Generate subtitle/transcript files from video or audio files using
local AI speech recognition with faster-whisper and local AI 
translation with NLLB-200.

This script is just a wrapper/interfacing layer around the models
to make it easier to use. The heavy lifting is done by ffmpeg,
faster-whisper and NLLB-200.
"""

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def signal_handler(sig, frame):
    """Handle interrupt signal for clean shutdown."""
    print("\n\n⚠️  Interrupted! Cleaning up...")
    # Force garbage collection to free model memory
    import gc
    gc.collect()
    print("Cleanup complete. Exiting.")
    sys.exit(0)


# Register signal handler for CTRL+C
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster-whisper is not installed.")
    print("Please install it with: pip install faster-whisper")
    sys.exit(1)


# NLLB language codes mapping (ISO 639-1 -> NLLB format)
# See: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "ar": "arb_Arab", "hi": "hin_Deva",
    "nl": "nld_Latn", "pl": "pol_Latn", "tr": "tur_Latn", "sv": "swe_Latn",
    "no": "nob_Latn", "da": "dan_Latn", "fi": "fin_Latn", "el": "ell_Grek",
    "he": "heb_Hebr", "th": "tha_Thai", "vi": "vie_Latn", "id": "ind_Latn",
    "bg": "bul_Cyrl", "uk": "ukr_Cyrl", "cs": "ces_Latn", "ro": "ron_Latn",
    "hu": "hun_Latn", "sk": "slk_Latn", "hr": "hrv_Latn", "sr": "srp_Cyrl",
    "sl": "slv_Latn", "et": "est_Latn", "lv": "lvs_Latn", "lt": "lit_Latn",
    "mk": "mkd_Cyrl", "sq": "als_Latn", "bs": "bos_Latn", "mt": "mlt_Latn",
    "is": "isl_Latn", "ga": "gle_Latn", "cy": "cym_Latn", "af": "afr_Latn",
    "sw": "swh_Latn", "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu",
    "ml": "mal_Mlym", "kn": "kan_Knda", "mr": "mar_Deva", "gu": "guj_Gujr",
    "pa": "pan_Guru", "ur": "urd_Arab", "fa": "pes_Arab", "ms": "zsm_Latn",
    "tl": "tgl_Latn", "my": "mya_Mymr", "km": "khm_Khmr", "lo": "lao_Laoo",
    "ne": "npi_Deva", "si": "sin_Sinh", "ka": "kat_Geor", "hy": "hye_Armn",
    "az": "azj_Latn", "kk": "kaz_Cyrl", "uz": "uzn_Latn", "mn": "khk_Cyrl",
}

# Language names for display
LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi",
    "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "sv": "Swedish",
    "no": "Norwegian", "da": "Danish", "fi": "Finnish", "el": "Greek",
    "he": "Hebrew", "th": "Thai", "vi": "Vietnamese", "id": "Indonesian",
    "bg": "Bulgarian", "uk": "Ukrainian", "cs": "Czech", "ro": "Romanian",
    "hu": "Hungarian", "sk": "Slovak", "hr": "Croatian", "sr": "Serbian",
    "sl": "Slovenian", "et": "Estonian", "lv": "Latvian", "lt": "Lithuanian",
    "mk": "Macedonian", "sq": "Albanian", "bs": "Bosnian", "mt": "Maltese",
    "is": "Icelandic", "ga": "Irish", "cy": "Welsh", "af": "Afrikaans",
    "sw": "Swahili", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "ml": "Malayalam", "kn": "Kannada", "mr": "Marathi", "gu": "Gujarati",
    "pa": "Punjabi", "ur": "Urdu", "fa": "Persian", "ms": "Malay",
    "tl": "Filipino", "my": "Myanmar", "km": "Khmer", "lo": "Lao",
    "ne": "Nepali", "si": "Sinhala", "ka": "Georgian", "hy": "Armenian",
    "az": "Azerbaijani", "kk": "Kazakh", "uz": "Uzbek", "mn": "Mongolian",
}


def print_supported_languages():
    """Print a table of all supported languages."""
    print("\n" + "="*70)
    print("SUPPORTED LANGUAGES (Whisper + NLLB-200)")
    print("="*70)
    print("\nUse these codes with --language, --translate-to, or --translate-via-english-to\n")
    
    # Sort by language name
    sorted_codes = sorted(LANGUAGE_NAMES.keys(), key=lambda x: LANGUAGE_NAMES[x])
    
    # Print in 3 columns
    print(f"{'Code':<6} {'Language':<15} {'Code':<6} {'Language':<15} {'Code':<6} {'Language':<15}")
    print("-"*70)
    
    # Create rows of 3
    for i in range(0, len(sorted_codes), 3):
        row = ""
        for j in range(3):
            if i + j < len(sorted_codes):
                code = sorted_codes[i + j]
                name = LANGUAGE_NAMES[code]
                row += f"{code:<6} {name:<15} "
        print(row)
    
    print("\n" + "="*70)
    print(f"Total: {len(LANGUAGE_NAMES)} languages supported")
    print("="*70 + "\n")



def translate_segments(segments: list, source_lang: str, target_lang: str, 
                        translation_model: str = "small", verbose: bool = True) -> list:
    """
    Translate transcript segments using NLLB-200 model.
    
    Args:
        segments: List of transcript segments with text
        source_lang: Source language code (ISO 639-1, e.g., 'ja')
        target_lang: Target language code (ISO 639-1, e.g., 'bg')
        translation_model: Model size - 'small', 'medium', or 'large'
        verbose: If True, print segments as they are translated
    
    Returns:
        List of segments with translated text
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    except ImportError:
        print("Error: transformers is not installed.")
        print("Please install it with: pip install transformers sentencepiece")
        sys.exit(1)
    
    # Map language codes to NLLB format
    src_nllb = NLLB_LANGUAGE_CODES.get(source_lang)
    tgt_nllb = NLLB_LANGUAGE_CODES.get(target_lang)
    
    if not src_nllb:
        print(f"Warning: Unknown source language '{source_lang}', using English")
        src_nllb = "eng_Latn"
    
    if not tgt_nllb:
        print(f"Error: Unknown target language '{target_lang}'")
        print(f"Supported languages: {', '.join(sorted(NLLB_LANGUAGE_CODES.keys()))}")
        sys.exit(1)
    
    # Select model based on size
    translation_models = {
        "small": "facebook/nllb-200-distilled-600M",      # ~2.3GB, fastest
        "medium": "facebook/nllb-200-distilled-1.3B",     # ~5GB, balanced
        "large": "facebook/nllb-200-3.3B",                # ~13GB, best quality
    }
    model_name = translation_models.get(translation_model, translation_models["small"])
    
    print(f"\nLoading NLLB-200 translation model ({translation_model})...")
    print(f"Translating: {source_lang} ({src_nllb}) → {target_lang} ({tgt_nllb})")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=src_nllb,
        tgt_lang=tgt_nllb,
        max_length=512
    )
    
    print(f"Translating {len(segments)} segments...")
    
    translated_segments = []
    for i, segment in enumerate(segments):
        # Translate text
        result = translator(segment["text"])
        translated_text = result[0]["translation_text"]
        
        translated_segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": translated_text
        })
        
        # Print progress
        if verbose:
            print(f"  [{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}] {translated_text}")
        elif (i + 1) % 10 == 0 or i == len(segments) - 1:
            print(f"  Translated {i + 1}/{len(segments)} segments...")
    
    print(f"Translation complete.")
    return translated_segments



def check_ffmpeg():
    """Check if FFmpeg is installed and available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_audio(video_path: str, audio_path: str) -> bool:
    """
    Extract audio from video file using FFmpeg.
    
    Args:
        video_path: Path to the input video file
        audio_path: Path for the output audio file (WAV format)
    
    Returns:
        True if extraction was successful, False otherwise
    """
    print(f"Extracting audio from: {video_path}")
    
    try:
        command = [
            "ffmpeg",
            "-i", video_path,
            "-vn",                      # No video
            "-acodec", "pcm_s16le",     # PCM 16-bit little-endian
            "-ar", "16000",             # 16kHz sample rate
            "-ac", "1",                 # Mono channel
            "-y",                       # Overwrite output file
            audio_path
        ]
        
        subprocess.run(command, check=True, stderr=subprocess.PIPE)
        print("Audio extraction complete.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed to extract audio: {e.stderr.decode()}")

        return False
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False


def get_potential_output_paths(input_path: Path, args) -> list:
    """
    Determine the list of output files that will be generated based on arguments.
    """
    paths = []
    
    # Determine extension
    format_ext = f".{args.format}"
    
    # Case 1: Explicit output path (single file)
    if args.output:
        return [Path(args.output)]
    
    # Case 2: Multi-language translation (--translate-to or --translate-via-english-to)
    target_langs = []
    if args.translate_to:
        target_langs = args.translate_to.split(",")
    elif args.translate_via_english_to:
        target_langs = args.translate_via_english_to.split(",")
        
    if target_langs:
        for lang in target_langs:
            lang = lang.strip()
            paths.append(input_path.with_suffix(f".{lang}{format_ext}"))
        
        # When translating to specific targets, we usually also generate the base transcription
        # unless it's a direct translate task (which this script typically does as separate steps).
        # The main logic always runs transcribe_audio first and saves it.
        # So we should ALSO expect the base file.
        paths.append(input_path.with_suffix(format_ext))
        return paths
        
    # Case 3: Single translation (--translate defaults to English)
    if args.translate:
        # Default suffix for --translate is .en
        paths.append(input_path.with_suffix(f".en{format_ext}"))
        # Does --translate ALSO save the original language transcription?
        # If task="translate", Whisper translates directly. 
        # If so, it produces ONLY the translated text (English).
        # So we do NOT add the base path here.
        return paths

    # Case 4: Default Transcription
    # Logic in main: output_path = input_path.with_suffix(format_ext)
    paths.append(input_path.with_suffix(format_ext))
    
    return paths


def detect_fps(input_path: str) -> float:
    """Detect frames per second using ffprobe."""
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        fps_str = result.stdout.strip()
        
        # Handle formats like "24/1" or "30000/1001"
        if "/" in fps_str:
            num, den = map(float, fps_str.split("/"))
            return num / den if den != 0 else 25.0
        return float(fps_str)
    except (subprocess.CalledProcessError, ValueError, IndexError):
        # Fallback to default if detection fails (e.g., audio file)
        return 25.0


def check_existing_subtitles(input_path: str) -> bool:
    """Check if the video file contains any subtitle streams."""
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "s",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        input_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return len(result.stdout.strip()) > 0
    except subprocess.CalledProcessError:
        return False


def extract_subtitle(input_path: str, output_path: str) -> bool:
    """Extract the first subtitle stream to a file."""
    command = [
        "ffmpeg",
        "-i", input_path,
        "-map", "0:s:0",
        "-y",
        output_path
    ]
    try:
        subprocess.run(command, check=True, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        # print(f"FFmpeg stderr: {e.stderr.decode()}")
        return False


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def transcribe_audio(audio_path: str, model_size: str = "medium", 
                     device: str = "auto", language: str = None,
                     task: str = "transcribe", verbose: bool = True) -> tuple:
    """
    Transcribe or translate audio using faster-whisper.
    
    Args:
        audio_path: Path to the audio file
        model_size: Whisper model size (tiny, base, small, medium, large-v3)
        device: Device to use (auto, cpu, cuda)
        language: Language code (e.g., 'en', 'es') or None for auto-detection
        task: "transcribe" for same-language transcription, "translate" to translate to English
        verbose: If True, print segments as they are generated
    
    Returns:
        Tuple of (transcript segments list, audio duration in seconds, transcription time in seconds)
    """
    import platform
    
    print(f"Loading {model_size} model...")
    
    # Auto-detect best device and compute type
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
                print(f"Detected NVIDIA GPU - using CUDA with float16")
            else:
                device = "cpu"
                # Check for Apple Silicon
                if platform.system() == "Darwin" and platform.machine() == "arm64":
                    compute_type = "int8"
                    print(f"Detected Apple Silicon - using CPU with int8")
                else:
                    # Intel/AMD CPU
                    compute_type = "int8"
                    if verbose:
                        print(f"Detected x86 CPU - using CPU with int8")
        except ImportError:
            # torch not installed, use CPU
            device = "cpu"
            compute_type = "int8"
            if verbose:
                print(f"Using CPU with int8 (PyTorch not available for GPU detection)")
    elif device == "cuda":
        compute_type = "float16"
    else:
        compute_type = "int8"
    
    if verbose:
        print(f"Device: {device}, Compute type: {compute_type}")
    
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    if verbose:
        if task == "translate":
            print("Translating audio to English (this may take a while)...")
        else:
            print("Transcribing audio (this may take a while)...")
    
    # Start timing
    start_time = time.time()
    
    # Transcribe/translate with word-level timestamps for better accuracy
    segments_generator, info = model.transcribe(
        audio_path,
        language=language,
        task=task,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,  # Filter out non-speech segments
    )
    
    if verbose:
        if language is None:
            print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        if task == "translate":
            print(f"Translating from {info.language} to English...")
    
    # Collect all segments
    transcript = []
    audio_duration = 0
    for segment in segments_generator:
        transcript.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
        audio_duration = max(audio_duration, segment.end)
        # Print progress
        if verbose:
            print(f"  [{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}] {segment.text.strip()}")
    
    # End timing
    transcription_time = time.time() - start_time
    
    task_name = "Translation" if task == "translate" else "Transcription"
    if verbose:
        print(f"\n{task_name} complete. {len(transcript)} segments found.")
    return transcript, audio_duration, transcription_time


def generate_output(segments: list, output_path: str, format: str = "srt", 
                    fps: float = 25.0, max_line_length: int = 42):
    """
    Generate subtitle/transcript file in the specified format.
    
    Args:
        segments: List of transcript segments
        output_path: Path for the output file
        format: Output format (srt, vtt, ass, sub, txt, json)
        fps: Frames per second (only used for SUB format)
        max_line_length: Maximum characters per line (for readability)
    """
    format_handlers = {
        "srt": generate_srt,
        "vtt": generate_vtt,
        "ass": generate_ass,
        "sub": lambda s, p, ml: generate_sub(s, p, fps), # Use the fps parameter
        "txt": generate_txt,
        "json": generate_json,
    }
    
    handler = format_handlers.get(format.lower())
    if handler:
        # All handlers now take segments, output_path, and max_line_length (or fps for sub)
        # The lambda for 'sub' already captures 'fps' from the generate_output scope.
        # So, we can just call the handler with the common arguments.
        handler(segments, output_path, max_line_length)
    else:
        raise ValueError(f"Unsupported format: {format}")


def generate_srt(segments: list, output_path: str, max_line_length: int = 42):
    """Generate SRT subtitle file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            f.write(f"{i}\n")
            start_ts = format_timestamp(segment["start"])
            end_ts = format_timestamp(segment["end"])
            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{_split_text(segment['text'], max_line_length)}\n")
            f.write("\n")


def generate_vtt(segments: list, output_path: str, max_line_length: int = 42):
    """Generate WebVTT subtitle file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, segment in enumerate(segments, start=1):
            # VTT uses period for milliseconds, not comma
            start_ts = format_timestamp(segment["start"]).replace(",", ".")
            end_ts = format_timestamp(segment["end"]).replace(",", ".")
            f.write(f"{i}\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{_split_text(segment['text'], max_line_length)}\n")
            f.write("\n")


def generate_ass(segments: list, output_path: str, max_line_length: int = 42):
    """Generate ASS (Advanced SubStation Alpha) subtitle file."""
    with open(output_path, "w", encoding="utf-8") as f:
        # ASS header
        f.write("[Script Info]\n")
        f.write("Title: Auto-generated subtitles\n")
        f.write("ScriptType: v4.00+\n")
        f.write("Collisions: Normal\n")
        f.write("PlayDepth: 0\n\n")
        
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write("Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1\n\n")
        
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for segment in segments:
            start = _format_ass_time(segment["start"])
            end = _format_ass_time(segment["end"])
            text = segment["text"].replace("\n", "\\N")
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")


def generate_sub(segments: list, output_path: str, fps: float = 25.0):
    """Generate SUB (MicroDVD) subtitle file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            start_frame = int(segment["start"] * fps)
            end_frame = int(segment["end"] * fps)
            text = segment["text"].replace("\n", "|")
            f.write(f"{{{start_frame}}}{{{end_frame}}}{text}\n")


def generate_txt(segments: list, output_path: str, max_line_length: int = 42):
    """Generate plain text transcript (no timestamps)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            f.write(f"{segment['text']}\n")


def generate_json(segments: list, output_path: str, max_line_length: int = 42):
    """Generate JSON transcript file."""
    import json
    output = {
        "segments": [
            {
                "id": i,
                "start": segment["start"],
                "end": segment["end"],
                "start_formatted": format_timestamp(segment["start"]),
                "end_formatted": format_timestamp(segment["end"]),
                "text": segment["text"]
            }
            for i, segment in enumerate(segments, start=1)
        ],
        "total_segments": len(segments),
        "total_duration": segments[-1]["end"] if segments else 0
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def _format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp format (H:MM:SS.cc)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def _split_text(text: str, max_line_length: int = 42) -> str:
    """Split long text lines for readability."""
    if len(text) > max_line_length:
        mid = len(text) // 2
        split_pos = text.rfind(" ", 0, mid + 10)
        if split_pos == -1 or split_pos < mid - 10:
            split_pos = text.find(" ", mid)
        if split_pos != -1:
            text = text[:split_pos] + "\n" + text[split_pos + 1:]
    return text

def main():
    """Main entry point for the auto-subtitles tool."""
    parser = argparse.ArgumentParser(
        description="Generate SRT subtitles/transcripts from video or audio files using local AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription
  %(prog)s sample-video.mp4                           # Transcribe in original language
  %(prog)s sample-audio.mp3                           # Works with audio too
  
  # Whisper model selection (transcription quality)
  %(prog)s sample-video.mp4 -m tiny                   # Fast, lower accuracy
  %(prog)s sample-video.mp4 -m large-v3               # Slow, best accuracy
  %(prog)s sample-video.mp4 -m small --language en    # Balanced + specify language
  
  # Translate to English only (Whisper built-in)
  %(prog)s sample-video.mp4 --translate               # Any language to English
  
  # Translate to any language (NLLB-200)
  %(prog)s sample-video.mp4 --translate-to fr         # Translate to French
  %(prog)s sample-video.mp4 --translate-to en,fr,ja   # Multiple languages at once
  %(prog)s sample-video.mp4 --translate-via-english-to fr  # Via English for better accuracy
  
  # NLLB translation model selection (translation quality)
  %(prog)s sample-video.mp4 --translate-to fr --translation-model small   # Fast
  %(prog)s sample-video.mp4 --translate-to fr --translation-model large   # Best quality
  
  # Combine Whisper + NLLB model sizes
  %(prog)s sample-video.mp4 -m large-v3 --translate-to fr --translation-model large
  
  # Output formats
  %(prog)s sample-video.mp4 --format vtt              # WebVTT for web
  %(prog)s sample-video.mp4 --format txt              # Plain text transcript
  
Whisper transcription models (-m, --model):
  tiny     - ~75MB,  fastest, lower accuracy
  base     - ~145MB, fast, good accuracy  
  small    - ~480MB, balanced speed/accuracy
  medium   - ~1.5GB, high accuracy [default]
  large-v3 - ~3GB,   best accuracy

NLLB translation models (--translation-model):
  small    - ~2.3GB, fast translations [default]
  medium   - ~5GB,   better quality
  large    - ~13GB,  best quality
        """
    )
    
    parser.add_argument(
        "input",
        nargs="?",
        metavar="FILE",
        help="Path to video or audio file (MP4, MKV, MP3, WAV, etc.)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output SRT file path (default: same as input with .srt extension)"
    )
    
    parser.add_argument(
        "-m", "--model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: medium)"
    )
    
    parser.add_argument(
        "-l", "--language",
        metavar="LANG",
        help="Language code (e.g., 'en', 'es', 'fr'). Auto-detected if not specified."
    )
    
    parser.add_argument(
        "-d", "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference (default: auto)"
    )
    
    parser.add_argument(
        "-b", "--benchmark",
        action="store_true",
        help="Run full benchmark: test all Whisper + NLLB model sizes and compare performance"
    )
    
    parser.add_argument(
        "--benchmark-transcribe-only",
        action="store_true",
        help="Run Whisper-only benchmark: test all transcription model sizes (no translation)"
    )
    
    parser.add_argument(
        "-t", "--translate",
        action="store_true",
        help="Translate to English: produces English subtitles regardless of the audio's original language (uses Whisper)"
    )
    
    parser.add_argument(
        "--translate-to", "--translate-directly-to",
        dest="translate_to",
        metavar="LANG",
        help="Translate directly to target language (e.g., 'fr', 'es'). Uses NLLB-200 for direct translation."
    )
    
    parser.add_argument(
        "--translate-via-english-to",
        dest="translate_via_english",
        metavar="LANG",
        help="Translate via English: Whisper transcribes to English, then NLLB translates to target language."
    )
    
    parser.add_argument(
        "--translation-model",
        default="small",
        choices=["small", "medium", "large"],
        help="NLLB translation model size: small (~2.3GB), medium (~5GB), large (~13GB) (default: small)"
    )
    
    parser.add_argument(
        "-f", "--format",
        default="srt",
        choices=["srt", "vtt", "ass", "sub", "txt", "json"],
        help="Output format: srt, vtt, ass, sub (MicroDVD), txt (plain text), json (default: srt)"
    )
    
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frames per second for SUB format. Ignored for other formats. (default: auto-detect from video, or 25.0)"
    )
    
    parser.add_argument(
        "--list-all-supported-languages",
        action="store_true",
        help="List all supported language codes and exit"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress console output of generated subtitles"
    )

    args = parser.parse_args()
    
    # Handle --list-all-supported-languages
    if args.list_all_supported_languages:
        print_supported_languages()
        sys.exit(0)
    
    # Require FILE argument for all other operations
    if not args.input:
        parser.error("the following arguments are required: FILE")
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    # Supported formats (video and audio)
    supported_formats = [
        # Video
        ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v",
        ".mpg", ".mpeg", ".3gp", ".ts", ".mts", ".m2ts",
        # Audio
        ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus"
    ]
    
    if input_path.suffix.lower() not in supported_formats:
        print(f"Warning: Unknown format '{input_path.suffix}'. Attempting anyway...")
    
    # Check FFmpeg
    if not check_ffmpeg():
        print("Error: FFmpeg is not installed or not in PATH.")
        print("Please install FFmpeg: https://ffmpeg.org/download.html")
        sys.exit(1)
    
    # Get system info
    import platform
    cpu_info = platform.processor() or platform.machine()
    system_info = f"{platform.system()} {platform.machine()}"

    # Check for existing subtitles
    if not args.benchmark and not args.benchmark_transcribe_only:
        print(f"Input file: {input_path}")
    
        # Check for EXISTING EXTERNAL subtitle files first
        potential_outputs = get_potential_output_paths(input_path, args)
        existing_files = [p for p in potential_outputs if p.exists()]
        
        if existing_files:
            print(f"\n⚠️  External subtitle file(s) found:")
            for p in existing_files:
                print(f"   - {p.name}")
            
            print(f"\n   DANGER: Generating new subtitles will OVERWRITE these files.")
            response = input(f"   Do you want to use the existing file(s) instead? (y/n): ").lower().strip()
            
            if response == 'y':
                print(f"✅ Using existing files. Exiting.")
                sys.exit(0)
            else:
                print(f"⚠️  Will overwrite existing files...")

        # Check for INTERNAL subtitle streams
        if check_existing_subtitles(str(input_path)):
            print(f"\n⚠️  Subtitle stream detected in the input file!")
            response = input("   Do you want to extract existing subtitles instead of generating new ones? (y/n): ").strip().lower()
            
            if response == 'y':
                # Determine output path
                format_ext = f".{args.format}"
                if args.output:
                    output_path = Path(args.output)
                else:
                    output_path = input_path.with_suffix(format_ext)
                
                print(f"   Extracting subtitles to {output_path}...")
                if extract_subtitle(str(input_path), str(output_path)):
                    print(f"   ✅ Extraction complete.")
                    sys.exit(0)
                else:
                    print(f"   ❌ Extraction failed. Continuing with generation...")


    
    # Create temporary directory for audio
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, "audio.wav")
        
        # Step 1: Extract audio (once for all models)
        if not extract_audio(str(input_path), audio_path):
            print("Failed to extract audio from file.")
            sys.exit(1)
        
        if args.benchmark:
            # Full benchmark mode: Whisper + NLLB models
            run_full_benchmark(input_path, audio_path, args, system_info, cpu_info)
        elif args.benchmark_transcribe_only:
            # Whisper-only benchmark mode
            run_whisper_benchmark(input_path, audio_path, args, system_info, cpu_info)
        else:
            # Normal mode: run single model
            run_single_model(input_path, audio_path, args, system_info, cpu_info)


def run_single_model(input_path, audio_path, args, system_info, cpu_info):
    """Run transcription with a single model."""
    format_ext = f".{args.format}"
    
    # Determine Whisper task (transcribe in original language, or translate to English)
    # If using --translate-via-english-to, we need English first from Whisper
    if args.translate or args.translate_via_english:
        whisper_task = "translate"
    else:
        whisper_task = "transcribe"
    
    # Transcribe/translate audio with Whisper
    try:
        segments, audio_duration, transcription_time = transcribe_audio(
            audio_path,
            model_size=args.model,
            device=args.device,
            language=args.language,
            task=whisper_task,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Transcription failed: {e}")
        sys.exit(1)
    
    if not segments:
        print("No speech detected in the file.")
        sys.exit(0)
    
    # Store detected source language for NLLB
    detected_lang = args.language or "en"  # Default to English if not detected
    
    # Determine FPS if format is SUB
    fps_val = 25.0
    if args.format.lower() == "sub":
        if args.fps is not None:
            fps_val = args.fps
        else:
            print("   Detecting framerate...")
            fps_val = detect_fps(str(input_path))
            print(f"   Detected/Default FPS: {fps_val:.3f}")

    # Handle different translation scenarios
    outputs_generated = []
    
    # Case 1: Simple transcription or Whisper translate to English
    if not args.translate_to and not args.translate_via_english:
        if args.output:
            output_path = Path(args.output)
        else:
            if args.translate:
                output_path = input_path.with_suffix(f".en{format_ext}")
            else:
                output_path = input_path.with_suffix(format_ext)
        
        generate_output(segments, str(output_path), format=args.format, fps=fps_val)
        outputs_generated.append((output_path, "en" if args.translate else detected_lang))
    
    # Case 2: Direct translation to target language(s) using NLLB
    elif args.translate_to:
        target_langs = [lang.strip() for lang in args.translate_to.split(",")]
        
        for target_lang in target_langs:
            print(f"\n{'─'*40}")
            print(f"Translating to: {target_lang}")
            print(f"{'─'*40}")
            
            # Translate segments using NLLB
            translated_segments = translate_segments(segments, detected_lang, target_lang, args.translation_model, verbose=not args.quiet)
            
            # Generate output
            output_path = input_path.with_suffix(f".{target_lang}{format_ext}")
            generate_output(translated_segments, str(output_path), format=args.format, fps=fps_val)
            outputs_generated.append((output_path, target_lang))
    
    # Case 3: Translate via English using NLLB
    elif args.translate_via_english:
        # segments are already in English from Whisper translate
        target_langs = [lang.strip() for lang in args.translate_via_english.split(",")]
        
        for target_lang in target_langs:
            if target_lang == "en":
                # Already in English, just save
                output_path = input_path.with_suffix(f".en{format_ext}")
                generate_output(segments, str(output_path), format=args.format, fps=args.fps)
            else:
                print(f"\n{'─'*40}")
                print(f"Translating to: {target_lang}")
                print(f"{'─'*40}")
                
                # Translate from English using NLLB
                translated_segments = translate_segments(segments, "en", target_lang, args.translation_model, verbose=not args.quiet)
                output_path = input_path.with_suffix(f".{target_lang}{format_ext}")
                generate_output(translated_segments, str(output_path), format=args.format, fps=args.fps)
            
            outputs_generated.append((output_path, target_lang))
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Done! Generated {len(outputs_generated)} subtitle file(s):")
    for path, lang in outputs_generated:
        print(f"   📄 {path} ({lang})")
    print(f"{'='*60}")
    
    print(f"\n📊 Performance Summary:")
    print(f"   Format:              {args.format.upper()}")
    lang_codes = ", ".join([lang for _, lang in outputs_generated])
    print(f"   Languages:           {lang_codes}")
    print(f"   Model:               {args.model}")
    print(f"   System:              {system_info}")
    print(f"   CPU:                 {cpu_info}")
    print(f"   Audio duration:      {format_timestamp(audio_duration)} ({audio_duration:.1f}s)")
    print(f"   Processing time:     {format_timestamp(transcription_time)} ({transcription_time:.1f}s)")
    
    if transcription_time > 0:
        speed_ratio = audio_duration / transcription_time
        print(f"   Speed:               {speed_ratio:.2f}x realtime")
        
        if speed_ratio >= 1:
            print(f"   ✅ Faster than realtime!")
        else:
            print(f"   ⏱️  {1/speed_ratio:.1f}x slower than realtime")



def print_nllb_benchmark_summary(nllb_results, source_segment_count):
    """Print summary of NLLB benchmark results."""
    print(f"\n\n{'='*80}")
    print(f"📊 NLLB TRANSLATION BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"Source: {source_segment_count} segments | Target language: French (fr)")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<15} {'Download':>12} {'Time':>12} {'Status':<10}")
    print(f"{'-'*15} {'-'*12} {'-'*12} {'-'*10}")
    
    model_sizes = {"small": "~2.3GB", "medium": "~5GB", "large": "~13GB"}
    for r in nllb_results:
        time_str = f"{r['translation_time']:.1f}s"
        print(f"{r['model']:<15} {model_sizes[r['model']]:>12} {time_str:>12} {r['status']:<10}")
    
    print(f"\n{'='*80}")
    
    if nllb_results:
        successful = [r for r in nllb_results if r["status"] == "✅"]
        if successful:
            fastest = min(successful, key=lambda x: x["translation_time"])
            print(f"\n🏆 Fastest NLLB:  {fastest['model']} ({fastest['translation_time']:.1f}s)")
            print(f"\n💡 Recommendation:")
            print(f"   • For speed:   Use 'small' (default)")
            print(f"   • For quality: Use 'large' (best translations)")
    
    print(f"\n📁 Output files saved with NLLB model suffix (e.g., video.nllb-small.fr.srt)")


def print_whisper_benchmark_summary(results, system_info, cpu_info, input_path):
    """Print summary of Whisper benchmark results."""
    # Print benchmark summary
    print(f"\n\n{'='*80}")
    print(f"📊 WHISPER BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"System: {system_info} | CPU: {cpu_info}")
    print(f"File:   {input_path.name}")
    if results and results[0]["audio_duration"] > 0:
        print(f"Duration: {format_timestamp(results[0]['audio_duration'])} ({results[0]['audio_duration']:.1f}s)")
    print(f"{'='*80}\n")
    
    # Table header
    print(f"{'Model':<12} {'Time':>12} {'Speed':>14} {'Segments':>10} {'Status':<10}")
    print(f"{'-'*12} {'-'*12} {'-'*14} {'-'*10} {'-'*10}")
    
    for r in results:
        time_str = format_timestamp(r["transcription_time"]) if r["transcription_time"] > 0 else "N/A"
        speed_str = f"{r['speed_ratio']:.2f}x" if r["speed_ratio"] > 0 else "N/A"
        print(f"{r['model']:<12} {time_str:>12} {speed_str:>14} {r['segments']:>10} {r['status']:<10}")
    
    print(f"\n{'='*80}")
    
    # Find fastest and recommend best
    successful = [r for r in results if r["speed_ratio"] > 0]
    if successful:
        fastest = max(successful, key=lambda x: x["speed_ratio"])
        fewest_segments = min(successful, key=lambda x: x["segments"])
        
        print(f"\n🏆 Fastest:              {fastest['model']} ({fastest['speed_ratio']:.2f}x realtime)")
        print(f"🎯 Best grouping:        {fewest_segments['model']} ({fewest_segments['segments']} segments)")
        print(f"                         (fewer segments = more natural sentence grouping)")
        
        # Recommend based on balance
        print(f"\n💡 Recommendation:")
        print(f"   • For speed:          Use 'tiny' or 'base'")
        print(f"   • For movies/TV:      Use 'medium' or 'large-v3' (better sentence grouping)")
        print(f"   • For lectures:       Use 'small' (good balance of speed and precision)")
    
    print(f"\n📁 Output files saved with model suffix (e.g., video.small.srt)")


def run_whisper_benchmark(input_path, audio_path, args, system_info, cpu_info, print_summary=True):
    """Run benchmark across all Whisper model sizes (transcription only)."""
    models = ["tiny", "base", "small", "medium", "large-v3"]
    results = []
    
    print(f"\n{'='*80}")
    print(f"🏁 WHISPER BENCHMARK - Testing transcription models")
    print(f"{'='*80}")
    print(f"   System: {system_info}")
    print(f"   CPU:    {cpu_info}")
    print(f"   File:   {input_path.name}")
    print(f"{'='*80}\n")
    
    for model in models:
        print(f"\n{'─'*40}")
        print(f"Testing model: {model}")
        print(f"{'─'*40}")
        
        # Generate output path for this model
        output_path = input_path.with_suffix(f".{model}.srt")
        
        try:
            segments, audio_duration, transcription_time = transcribe_audio(
                audio_path,
                model_size=model,
                device=args.device,
                language=args.language,
                verbose=not args.quiet
            )
            
            if segments:
                generate_srt(segments, str(output_path))
                speed_ratio = audio_duration / transcription_time if transcription_time > 0 else 0
                results.append({
                    "model": model,
                    "audio_duration": audio_duration,
                    "transcription_time": transcription_time,
                    "speed_ratio": speed_ratio,
                    "segments": len(segments),
                    "segment_list": segments,  # Store actual segments for reuse
                    "output": output_path,
                    "status": "✅"
                })
                print(f"   ✅ Saved: {output_path}")
                print(f"   ⏱️  Time Elapsed: {format_timestamp(transcription_time)} ({transcription_time:.1f}s)")
            else:
                results.append({
                    "model": model,
                    "audio_duration": 0,
                    "transcription_time": transcription_time,
                    "speed_ratio": 0,
                    "segments": 0,
                    "segment_list": None,
                    "output": None,
                    "status": "⚠️ No speech"
                })
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                "model": model,
                "audio_duration": 0,
                "transcription_time": 0,
                "speed_ratio": 0,
                "segments": 0,
                "segment_list": None,
                "output": None,
                "status": f"❌ Error"
            })
    
    if print_summary:
        print_whisper_benchmark_summary(results, system_info, cpu_info, input_path)
        
    return results


def run_full_benchmark(input_path, audio_path, args, system_info, cpu_info):
    """Run full benchmark across all Whisper and NLLB model sizes."""
    import time as time_module
    
    print(f"\n{'='*80}")
    print(f"🏁 FULL BENCHMARK - Whisper + NLLB Models")
    print(f"{'='*80}")
    print(f"   System: {system_info}")
    print(f"   CPU:    {cpu_info}")
    print(f"   File:   {input_path.name}")
    print(f"{'='*80}\n")
    
    # First, run Whisper benchmark across all models
    print(f"\n{'─'*80}")
    print(f"Step 1: Run Whisper Transcription Benchmark")
    print(f"{'─'*80}")
    
    whisper_results = run_whisper_benchmark(input_path, audio_path, args, system_info, cpu_info, print_summary=False)
    
    # Find the best quality model result to use for NLLB benchmark
    # We prefer 'large-v3' > 'medium' > 'small' > 'base' > 'tiny'
    preferred_models = ["large-v3", "medium", "small", "base", "tiny"]
    segments = None
    used_model = None

    for model_name in preferred_models:
        for r in whisper_results:
            if r["model"] == model_name and r.get("segment_list"):
                segments = r["segment_list"]
                used_model = model_name
                break
        if segments:
            break
            
    print(f"\n{'─'*80}")
    print(f"Step 2: Preparing for NLLB Benchmark")
    print(f"{'─'*80}")
    
    if not segments:
        print("No successful transcription found from parsing step. Skipping NLLB benchmark.")
        return
        
    print(f"✅ Reusing source segments from Whisper '{used_model}' model for translation")
    print(f"   (Source contains {len(segments)} segments)")
    
    # Test NLLB translation models
    nllb_models = ["small", "medium", "large"]
    nllb_results = []
    
    print(f"\n{'─'*80}")
    print(f"Step 3: Test NLLB translation models (translating to French)")
    print(f"{'─'*80}")
    
    for nllb_model in nllb_models:
        print(f"\n{'─'*40}")
        print(f"Testing NLLB model: {nllb_model}")
        print(f"{'─'*40}")
        
        start_time = time_module.time()
        try:
            translated_segments = translate_segments(segments, "en", "fr", nllb_model, verbose=not args.quiet)
            translation_time = time_module.time() - start_time
            
            # Save output
            output_path = input_path.with_suffix(f".nllb-{nllb_model}.fr.srt")
            generate_srt(translated_segments, str(output_path))
            
            nllb_results.append({
                "model": nllb_model,
                "translation_time": translation_time,
                "segments": len(translated_segments),
                "output": output_path,
                "status": "✅"
            })
            print(f"   ✅ Saved: {output_path}")
            print(f"   ⏱️  Time Elapsed: {translation_time:.1f}s")
        except Exception as e:
            translation_time = time_module.time() - start_time
            print(f"   ❌ Failed: {e}")
            nllb_results.append({
                "model": nllb_model,
                "translation_time": translation_time,
                "segments": 0,
                "output": None,
                "status": "❌ Error"
            })
    
    print(f"\n\n{'='*80}")
    print(f"🏁 FINAL RESULTS")
    print(f"{'='*80}")

    # Print Whisper benchmark summary FIRST (matches execution order)
    print_whisper_benchmark_summary(whisper_results, system_info, cpu_info, input_path)

    # Print NLLB benchmark summary SECOND
    print_nllb_benchmark_summary(nllb_results, len(segments))


if __name__ == "__main__":
    main()


