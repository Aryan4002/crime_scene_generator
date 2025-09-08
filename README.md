🧠 Multi-Modal Crime Scene Interpreter

This project is a multi-modal system designed to interpret crime-related images and textual observations to generate contextual insights, similarity-based reasoning, and a narrated video summarizing the incident. It leverages image captioning (BLIP-2), semantic similarity (SBERT), and TTS + video generation.

📁 Project Structure
├── main.py                  # Main script to run the interpreter  
├── Facts.csv                # CSV file containing factual statements and reasoning  
├── crime_story.mp4          # Output video narration (generated)  
├── narration.mp3            # Audio file generated from narrative  
└── README.md                # This file  

✅ Prerequisites
📦 Python Libraries

Install required Python packages:

pip install transformers sentence-transformers pandas torch gtts moviepy pillow

🧠 Models Used

Salesforce/blip2-opt-2.7b via Hugging Face Transformers

all-MiniLM-L6-v2 via Sentence Transformers

Models will be automatically downloaded when the script is run for the first time.

🧾 Facts.csv

Ensure a Facts.csv file with the following columns:

fact: factual observation

reasoning: reasoning behind the fact

📸 Image Inputs

Use .jpg, .jpeg, or .png image formats.

🧰 Dependencies
📍 ImageMagick

Install from https://imagemagick.org
.
Ensure the correct path in the script:

mpy_config.change_settings({"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})
os.environ["IMAGEMAGICK_PATH"] = r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"


Adjust path based on your system configuration.

🔤 Font Configuration

Ensure arial.ttf or any .ttf font is accessible in your system for PIL usage. Example:

font = ImageFont.truetype("arial.ttf", 40)


Modify path if needed.

▶️ How to Run

Execute the main script:

python main.py


Input image paths and/or text separated by commas. Example:

image1.jpg, image2.jpg, There was blood on the floor


The system will:

Caption images

Match facts & reasoning

Print analysis

Generate a narrative story

Create a narrated video: crime_story.mp4

🎯 Output

crime_story.mp4: Narrated video summarizing the crime

narration.mp3: Audio narration file

Command-line output for each process step
