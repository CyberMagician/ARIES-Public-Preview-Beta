"""Real-time Audio Input/Output handling with PyAudio for Gemini Live API."""

import asyncio
import logging
from typing import Optional
import pyaudio
from config import AudioConfig

logger = logging.getLogger(__name__)

class AudioInputManager:
    """Manages real-time microphone input streaming (16kHz 16-bit PCM mono)."""

    def __init__(self, config: AudioConfig = AudioConfig()):
        self.config = config
        self.pyaudio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self, loop: asyncio.AbstractEventLoop):
        """Opens microphone stream and starts background audio capture."""
        self._loop = loop
        self.running = True

        def audio_callback(in_data, frame_count, time_info, status):
            if self.running and in_data and self._loop and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self.queue.put_nowait, in_data)
            return (None, pyaudio.paContinue)

        try:
            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.config.INPUT_CHANNELS,
                rate=self.config.INPUT_SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.config.INPUT_CHUNK_SIZE,
                stream_callback=audio_callback,
            )
            self.stream.start_stream()
            logger.info("Microphone stream started (16kHz PCM).")
        except Exception as e:
            logger.error(f"Failed to open microphone: {e}")
            raise

    async def read_chunk(self) -> bytes:
        """Asynchronously awaits the next chunk of raw PCM audio."""
        return await self.queue.get()

    def stop(self):
        """Stops and cleans up the input audio stream."""
        self.running = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.pyaudio.terminate()
        logger.info("Microphone stream closed.")


class AudioOutputManager:
    """Manages real-time audio playback (24kHz 16-bit PCM mono) with instant interruption support."""

    def __init__(self, config: AudioConfig = AudioConfig()):
        self.config = config
        self.pyaudio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.running = False
        self.playback_task: Optional[asyncio.Task] = None
        self.is_playing = False

    def start(self):
        """Opens speaker output stream and starts playback worker."""
        self.running = True
        try:
            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.config.OUTPUT_CHANNELS,
                rate=self.config.OUTPUT_SAMPLE_RATE,
                output=True,
                frames_per_buffer=self.config.OUTPUT_CHUNK_SIZE,
            )
            self.playback_task = asyncio.create_task(self._playback_loop())
            logger.info("Speaker output stream started (24kHz PCM).")
        except Exception as e:
            logger.error(f"Failed to open speaker stream: {e}")
            raise

    async def put_chunk(self, data: bytes):
        """Adds a chunk of 24kHz PCM audio to the playback queue."""
        if self.running and data:
            await self.queue.put(data)

    def interrupt(self):
        """Immediately flushes pending playback on user interruption."""
        cleared_count = 0
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break
        self.is_playing = False
        logger.info(f"Playback interrupted. Cleared {cleared_count} buffered audio chunks.")

    async def _playback_loop(self):
        """Continuously writes PCM audio chunks to the audio output stream."""
        while self.running:
            try:
                chunk = await self.queue.get()
                self.is_playing = True
                # Run blocking stream.write in a separate thread to keep asyncio loop responsive
                if self.stream and self.stream.is_active():
                    await asyncio.to_thread(self.stream.write, chunk)
                self.queue.task_done()
                if self.queue.empty():
                    self.is_playing = False
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error during audio playback: {e}")
                await asyncio.sleep(0.01)

    def stop(self):
        """Stops and cleans up playback."""
        self.running = False
        self.interrupt()
        if self.playback_task:
            self.playback_task.cancel()
            self.playback_task = None
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.pyaudio.terminate()
        logger.info("Speaker output stream closed.")
