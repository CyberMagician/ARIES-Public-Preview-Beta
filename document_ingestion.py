"""Zero-Weight Document & Image Ingestion Pipeline for ARIES Public Preview Beta.

Supports .txt, .md, .csv, .json, source code, .docx/.pptx/.xlsx (stdlib zipfile + XML),
.pdf (macOS PDFKit + Vision OCR via swift CLI fallback), and images (Vision OCR + base64).
Uses a SHA-256 content-addressable cache so repeat uploads are instantaneous.
"""

import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from config import ARIES_DATA_DIR

DOC_CACHE_PATH = os.path.join(ARIES_DATA_DIR, "doc_cache.json")


class DocumentIngestionEngine:
    def __init__(self, cache_path: str = DOC_CACHE_PATH):
        self.cache_path = cache_path
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                    return d if isinstance(d, dict) else {}
            except Exception:
                pass
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as fh:
                json.dump(self._cache, fh, ensure_ascii=False)
        except Exception:
            pass

    def ingest_bytes(self, filename: str, data: bytes) -> dict:
        sha = hashlib.sha256(data).hexdigest()
        ext = os.path.splitext(filename)[1].lower()

        if ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic"):
            b64 = base64.b64encode(data).decode("ascii")
            ocr_text = self._cache.get(sha, {}).get("text")
            if ocr_text is None:
                ocr_text = self._vision_ocr_image(data, ext)
                self._cache[sha] = {"filename": filename, "text": ocr_text}
                self._save_cache()
            return {
                "filename": filename,
                "kind": "image",
                "text": ocr_text,
                "image_base64": b64,
                "cached": False,
            }

        if sha in self._cache:
            return {
                "filename": filename,
                "kind": "document",
                "text": self._cache[sha]["text"],
                "cached": True,
            }

        if ext in (".docx", ".pptx", ".xlsx"):
            text = self._extract_ooxml(data, ext)
        elif ext == ".pdf":
            text = self._extract_pdf(data)
        else:
            text = data.decode("utf-8", errors="replace")

        self._cache[sha] = {"filename": filename, "text": text[:120_000]}
        self._save_cache()
        return {
            "filename": filename,
            "kind": "document",
            "text": text[:120_000],
            "cached": False,
        }

    @staticmethod
    def _extract_ooxml(data: bytes, ext: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            chunks = []
            with zipfile.ZipFile(tmp_path, "r") as zf:
                for name in sorted(zf.namelist()):
                    if name.endswith(".xml") and any(k in name for k in ("document", "slide", "sharedStrings", "sheet")):
                        raw = zf.read(name)
                        root = ET.fromstring(raw)
                        texts = [el.text.strip() for el in root.iter() if el.text and el.text.strip()]
                        if texts:
                            chunks.append(" ".join(texts))
            return "\n\n".join(chunks) if chunks else "No readable text found in Office document."
        except Exception as e:
            return f"Failed to parse {ext}: {e}"
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    @staticmethod
    def _extract_pdf(data: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            pdf_path = tmp.name
        swift_code = (
            "import Foundation\nimport PDFKit\n"
            f'if let doc = PDFDocument(url: URL(fileURLWithPath: "{pdf_path}")) {{\n'
            "  var out = \"\"\n"
            "  for i in 0..<doc.pageCount { out += (doc.page(at: i)?.string ?? \"\") + \"\\n\" }\n"
            "  print(out)\n}\n"
        )
        try:
            res = subprocess.run(["swift", "-"], input=swift_code, text=True, capture_output=True, timeout=15)
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            pass
        finally:
            try:
                os.unlink(pdf_path)
            except Exception:
                pass
        # Lightweight fallback text scan
        raw = data.decode("latin-1", errors="ignore")
        strings = re.findall(r"\(([^()]{3,200})\)", raw)
        return "\n".join(strings[:400]) if strings else "PDF text extraction unavailable."

    @staticmethod
    def _vision_ocr_image(data: bytes, ext: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            img_path = tmp.name
        swift_code = (
            "import Foundation\nimport Vision\nimport AppKit\n"
            f'if let img = NSImage(contentsOfFile: "{img_path}"), let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) {{\n'
            "  let req = VNRecognizeTextRequest()\n"
            "  req.recognitionLevel = .accurate\n"
            "  try? VNImageRequestHandler(cgImage: cg).perform([req])\n"
            "  let lines = (req.results ?? []).compactMap { $0.topCandidates(1).first?.string }\n"
            "  print(lines.joined(separator: \"\\n\"))\n}\n"
        )
        try:
            res = subprocess.run(["swift", "-"], input=swift_code, text=True, capture_output=True, timeout=12)
            if res.returncode == 0:
                return res.stdout.strip()
        except Exception:
            pass
        finally:
            try:
                os.unlink(img_path)
            except Exception:
                pass
        return ""


doc_ingestor = DocumentIngestionEngine()
