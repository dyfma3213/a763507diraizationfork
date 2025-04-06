
  </a> 
  </a>
  <a href="https://colab.research.google.com/github/MahmoudAshraf97/whisper-diarization/blob/main/Whisper_Transcription_%2B_NeMo_Diarization.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
  </a>
 
</p>




# 
open in colab  open
<br />
go to update log 9.30
<br />
paste

!apt-get update
<br />
!apt-get install -y libcudnn8 libcudnn8-dev
<br />
!pip install numpy==1.26.4
<br />
!pip install git+https://github.com/dyfma3213/78dcfaab51005aa703ee21375f81ed31bc248560whisperxfork.git ctranslate2==4.4.0
<br />
!pip install "nemo-toolkit[asr]>=2.dev"
<br />
!pip install --no-deps git+https://github.com/dyfma3213/e976d93demucs#egg=demucs
<br />
!pip install git+https://github.com/dyfma3213/5a0dd7fdeepmultilingualpunctuation.git
<br />
!pip install git+https://github.com/dyfma3213/c7cc7cectcforcedaligneruploadedUROMAN.git

## What is it
This repository combines Whisper ASR capabilities with Voice Activity Detection (VAD) and Speaker Embedding to identify the speaker for each sentence in the transcription generated by Whisper. First, the vocals are extracted from the audio to increase the speaker embedding accuracy, then the transcription is generated using Whisper, then the timestamps are corrected and aligned using WhisperX to help minimize diarization error due to time shift. The audio is then passed into MarbleNet for VAD and segmentation to exclude silences, TitaNet is then used to extract speaker embeddings to identify the speaker for each segment, the result is then associated with the timestamps generated by WhisperX to detect the speaker for each word based on timestamps and then realigned using punctuation models to compensate for minor time shifts.


WhisperX and NeMo parameters are coded into diarize.py and helpers.py, I will add the CLI arguments to change them later
## Installation
`FFMPEG` and `Cython` are needed as prerequisites to install the requirements
```
pip install cython
```
or
```
sudo apt update && sudo apt install cython3
```
```
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg

# on Windows using WinGet (https://github.com/microsoft/winget-cli)
winget install ffmpeg
```
```
pip install -r requirements.txt
```
## Usage 

```
python diarize.py -a AUDIO_FILE_NAME
```

If your system has enough VRAM (>=10GB), you can use `diarize_parallel.py` instead, the difference is that it runs NeMo in parallel with Whisper, this can be beneficial in some cases and the result is the same since the two models are nondependent on each other. This is still experimental, so expect errors and sharp edges. Your feedback is welcome.

## Command Line Options

- `-a AUDIO_FILE_NAME`: The name of the audio file to be processed
- `--no-stem`: Disables source separation
- `--whisper-model`: The model to be used for ASR, default is `medium.en`
- `--suppress_numerals`: Transcribes numbers in their pronounced letters instead of digits, improves alignment accuracy
- `--device`: Choose which device to use, defaults to "cuda" if available
- `--language`: Manually select language, useful if language detection failed
- `--batch-size`: Batch size for batched inference, reduce if you run out of memory, set to 0 for non-batched inference

## Known Limitations
- Overlapping speakers are yet to be addressed, a possible approach would be to separate the audio file and isolate only one speaker, then feed it into the pipeline but this will need much more computation
- There might be some errors, please raise an issue if you encounter any.

## Future Improvements
- Implement a maximum length per sentence for SRT

## Acknowledgements
Special Thanks for [@adamjonas](https://github.com/adamjonas) for supporting this project
This work is based on [OpenAI's Whisper](https://github.com/openai/whisper) , [Faster Whisper](https://github.com/guillaumekln/faster-whisper) , [Nvidia NeMo](https://github.com/NVIDIA/NeMo) , and [Facebook's Demucs](https://github.com/facebookresearch/demucs)
