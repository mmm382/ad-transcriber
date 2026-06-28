import os
import io
import subprocess
import tempfile
import glob

from flask import Flask, request, jsonify, send_file
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ── CORS — handle all preflight and response headers manually ──

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Expose-Headers"] = "X-Filename, Content-Disposition"
    return response


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Expose-Headers"] = "X-Filename, Content-Disposition"
        return response


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ── AD TRANSCRIBER ───────────────────────────────────────

@app.route("/transcribe", methods=["POST", "OPTIONS"])
def transcribe():
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.mp3")

            if request.is_json:
                url = (request.json.get("url") or "").strip()
                if not url:
                    return jsonify({"error": "No URL provided"}), 400

                video_template = os.path.join(tmpdir, "video.%(ext)s")
                result = subprocess.run(
                    [
                        "yt-dlp",
                        "-o", video_template,
                        "--no-playlist",
                        "--max-filesize", "200M",
                        "--no-warnings",
                        url,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                if result.returncode != 0:
                    return jsonify({
                        "error": "Could not download video. Make sure the ad has a video and the link is correct.",
                        "details": result.stderr[-500:] if result.stderr else "",
                    }), 400

                files = glob.glob(os.path.join(tmpdir, "video.*"))
                if not files:
                    return jsonify({"error": "No video found — this ad might be image-only."}), 400
                video_path = files[0]

            elif "file" in request.files:
                f = request.files["file"]
                video_path = os.path.join(tmpdir, "upload" + os.path.splitext(f.filename or "")[1])
                f.save(video_path)

            else:
                return jsonify({"error": "Send a JSON body with 'url' or upload a 'file'."}), 400

            result = subprocess.run(
                [
                    "ffmpeg", "-i", video_path,
                    "-vn", "-acodec", "libmp3lame", "-q:a", "4",
                    "-y", audio_path,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0 or not os.path.exists(audio_path):
                return jsonify({"error": "Failed to extract audio."}), 500

            size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            if size_mb > 25:
                return jsonify({"error": f"Audio is {size_mb:.1f}MB — Whisper limit is 25MB."}), 400

            with open(audio_path, "rb") as af:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=af,
                    response_format="verbose_json",
                )

            english_text = None
            detected_lang = getattr(transcript, "language", "unknown")

            if detected_lang != "english":
                with open(audio_path, "rb") as af:
                    translation = client.audio.translations.create(
                        model="whisper-1",
                        file=af,
                    )
                english_text = translation.text

            return jsonify({
                "original": transcript.text,
                "language": detected_lang,
                "english": english_text or transcript.text,
            })

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Timed out — video might be too long."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/translate", methods=["POST", "OPTIONS"])
def translate():
    try:
        data = request.json
        text = (data.get("text") or "").strip()
        target_lang = (data.get("language") or "").strip()

        if not text or not target_lang:
            return jsonify({"error": "Provide 'text' and 'language'."}), 400

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"Translate the following text to {target_lang}. Output only the translation, nothing else. Preserve the tone and style.",
                },
                {"role": "user", "content": text},
            ],
            temperature=0.3,
        )

        return jsonify({
            "translation": response.choices[0].message.content,
            "language": target_lang,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── MEDIA DOWNLOADER (Instagram, Facebook, etc.) ─────────

@app.route("/download-media", methods=["POST", "OPTIONS"])
def download_media():
    try:
        data = request.json
        url = (data.get("url") or "").strip()
        if not url:
            return jsonify({"error": "No URL provided"}), 400

        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    "yt-dlp",
                    "-o", os.path.join(tmpdir, "%(title).80s.%(ext)s"),
                    "--no-playlist",
                    "--max-filesize", "200M",
                    "--no-warnings",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                return jsonify({
                    "error": "Could not download from this URL. Make sure it's a public post.",
                    "details": result.stderr[-500:] if result.stderr else "",
                }), 400

            files = [
                f for f in os.listdir(tmpdir)
                if os.path.isfile(os.path.join(tmpdir, f))
            ]

            if not files:
                return jsonify({"error": "No media found at this URL."}), 400

            filepath = os.path.join(tmpdir, files[0])
            filename = files[0]

            filename = "".join(c for c in filename if c.isalnum() or c in "._- ").strip()
            if not filename:
                filename = "download.mp4"

            ext = os.path.splitext(filename)[1].lower()
            mime_types = {
                ".mp4": "video/mp4",
                ".webm": "video/webm",
                ".mkv": "video/x-matroska",
                ".mov": "video/quicktime",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }
            mime = mime_types.get(ext, "application/octet-stream")

            with open(filepath, "rb") as f:
                file_data = io.BytesIO(f.read())

            response = send_file(
                file_data,
                mimetype=mime,
                as_attachment=True,
                download_name=filename,
            )
            response.headers["X-Filename"] = filename
            return response

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Download timed out — try a shorter video."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
