
# 🧠 Multi-Modal Crime Scene Interpreter

This project is a multi-modal system designed to interpret crime-related images and textual observations to generate contextual insights, similarity-based reasoning, and a narrated video summarizing the incident. It uses image captioning (BLIP-2), semantic similarity (SBERT), and TTS + video generation.

---

## 📁 Project Structure

```
├── main.py                  # Main script to run the interpreter
├── Facts.csv                # CSV file containing factual statements and reasoning
├── crime_story.mp4          # Output video narration (generated)
├── narration.mp3            # Audio file generated from narrative
└── README.md                # This file
```

---

## ✅ Prerequisites

### 📦 Python Libraries

Install the following Python packages:

```bash
pip install transformers sentence-transformers pandas torch gtts moviepy pillow
```

### 🧠 Models Required

* `Salesforce/blip2-opt-2.7b` via Hugging Face Transformers
* `all-MiniLM-L6-v2` via Sentence Transformers

These will be downloaded automatically the first time you run the script.

### 🧾 CSV File

You must provide a `Facts.csv` file with the following columns:

* `fact`: factual observation
* `reasoning`: reasoning behind the fact

### 📸 Images

Ensure images are in `.jpg`, `.jpeg`, or `.png` formats.

---

## 🧰 Dependencies

### 📍 ImageMagick

* Download & install [ImageMagick](https://imagemagick.org/index.php)
* Ensure the executable path is correct in:

```python
mpy_config.change_settings({"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})
os.environ["IMAGEMAGICK_PATH"] = r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"
```

Change it based on your system if necessary.

### 🔤 Fonts

Ensure `arial.ttf` or any .ttf font file is accessible for PIL. Modify this line if needed:

```python
font = ImageFont.truetype("arial.ttf", 40)
```

---

## How to Run

1. Run the script:

```bash
python main.py
```

2. Input image paths and/or text separated by commas. Example:

```bash
image1.jpg, image2.jpg, There was blood on the floor
```

3. The system will:

   * Caption images
   * Find the most similar fact and reasoning
   * Print the analysis
   * Generate a crime story narrative
   * Produce a narrated video: `crime_story.mp4`

---

## Output

* `crime_story.mp4` — a narrated video constructed from text+image reasoning
* `narration.mp3` — audio narration of the crime story
* CLI outputs for each step

---

## 🧠 Authors

* **Rohan Raghav** – Full Stack Developer & Machine Learning Enthusiast

Feel free to modify this project to suit forensic, educational, or AI storytelling needs!

