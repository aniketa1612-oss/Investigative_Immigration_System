# ML integration boundary

The application now integrates local faster-whisper transcription, local MediaPipe facial landmarks, and local Ollama generation. The remaining research work is evaluation rather than claiming production accuracy: test each supported language, accent, dialect, code-switching pattern, recording condition, and demographic group. Each adapter must expose model identity, confidence, limitations, and whether its output is measured, interpreted, or generated.

