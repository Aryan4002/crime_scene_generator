import os
import cv2
import numpy as np
import spacy
import re
import random
from moviepy.editor import *
from rembg import remove
import nltk
from nltk.corpus import wordnet as wn

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Directories
IMAGE_DIR = os.path.join("Clips", "Image")
VIDEO_REMOVE_BG_DIR = os.path.join("Clips", "Video_RemoveBG")
VIDEO_KEEP_BG_DIR = os.path.join("Clips", "Video_KeepBG")
OBJECT_DIR = os.path.join("Clips", "Object")
TEMP_DIR = "TempFrames"
os.makedirs(TEMP_DIR, exist_ok=True)


def normalize_synonym(word):
    synsets = wn.synsets(word)
    if synsets:
        return synsets[0].lemmas()[0].name().lower()
    return word.lower()


def extract_keywords(prompt):
    doc = nlp(prompt)
    keywords = []
    seen = set()
    for token in doc:
        if token.pos_ in ["NOUN", "VERB", "ADP", "ADV"]:
            lemma = token.lemma_.lower()
            normalized = normalize_synonym(lemma)
            if normalized not in seen:
                keywords.append(normalized)
                seen.add(normalized)
    return keywords


def tokenize_filename(filename):
    name, _ = os.path.splitext(filename.lower())
    return re.split(r'[_\W]+', name)


def score_files(directory, keywords):
    scored = []
    for file in os.listdir(directory):
        tokens = set(tokenize_filename(file))  # e.g., {'shed', 'blood'}
        keyword_tokens = [set(kw.split('_')) for kw in keywords]  # e.g., [{'shed'}, {'blood'}, {'shed', 'blood'}]

        matches = []
        for i, kw_token_set in enumerate(keyword_tokens):
            if kw_token_set.issubset(tokens):
                matches.append(keywords[i])

        if matches:
            scored.append({
                "file": os.path.join(directory, file),
                "tokens": list(tokens),
                "matches": set(matches),
                "score": len(matches),
            })
    return sorted(scored, key=lambda x: x["score"], reverse=True)


def select_media_files(keywords):
    image_matches = score_files(IMAGE_DIR, keywords)
    video_remove_matches = score_files(VIDEO_REMOVE_BG_DIR, keywords)
    video_keep_matches = score_files(VIDEO_KEEP_BG_DIR, keywords)

    used_files = set()
    used_keywords = set()
    selected_images = []
    selected_videos_remove = []
    selected_videos_keep = []

    # Raw keywords for fallback
    raw_keywords = []
    for kw in keywords:
        raw_keywords.extend(kw.split('_'))
    raw_keywords = list(dict.fromkeys(raw_keywords))

    # 1) Select by normalized keywords in order
    for kw in keywords:
        if kw in used_keywords:
            continue
        # Video remove BG
        for match in video_remove_matches:
            if match["file"] not in used_files and kw in match["matches"]:
                selected_videos_remove.append(match["file"])
                used_files.add(match["file"])
                used_keywords.add(kw)
                break
        # Video keep BG
        for match in video_keep_matches:
            if match["file"] not in used_files and kw in match["matches"]:
                selected_videos_keep.append(match["file"])
                used_files.add(match["file"])
                used_keywords.add(kw)
                break
        # Image
        for match in image_matches:
            if match["file"] not in used_files and kw in match["matches"]:
                selected_images.append(match["file"])
                used_files.add(match["file"])
                used_keywords.add(kw)
                break

    # 2) Fallback with raw keywords
    remaining_keywords = [kw for kw in raw_keywords if kw not in used_keywords]

    for kw in remaining_keywords:
        for match in video_remove_matches:
            if match["file"] not in used_files and kw in match["tokens"]:
                selected_videos_remove.append(match["file"])
                used_files.add(match["file"])
                used_keywords.add(kw)
                break
        for match in video_keep_matches:
            if match["file"] not in used_files and kw in match["tokens"]:
                selected_videos_keep.append(match["file"])
                used_files.add(match["file"])
                used_keywords.add(kw)
                break
        for match in image_matches:
            if match["file"] not in used_files and kw in match["tokens"]:
                selected_images.append(match["file"])
                used_files.add(match["file"])
                used_keywords.add(kw)
                break

    return selected_images, selected_videos_remove, selected_videos_keep

def select_objects(keywords):
    matched = []
    used_keywords = set()
    for file in os.listdir(OBJECT_DIR):
        name = file.lower()
        for keyword in keywords:
            if keyword in name and keyword not in used_keywords:
                matched.append(os.path.join(OBJECT_DIR, file))
                used_keywords.add(keyword)
                break
    return matched


def place_objects(objects, duration, screen_size=(1280, 720)):
    placed_clips = []
    used_positions = []

    for obj_path in objects:
        img_clip = ImageClip(obj_path).set_duration(duration).resize(height=150)
        max_x = screen_size[0] - img_clip.w
        max_y = screen_size[1] - img_clip.h

        for _ in range(10):
            x = random.randint(0, int(max_x))
            y = random.randint(0, int(max_y))
            overlap = any(abs(x - ux) < 100 and abs(y - uy) < 100 for ux, uy in used_positions)
            if not overlap:
                used_positions.append((x, y))
                img_clip = img_clip.set_position((x, y))
                placed_clips.append(img_clip)
                break
    return placed_clips


def remove_video_bg(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        removed = remove(frame_rgb)
        frame_clip = ImageClip(removed).set_duration(1 / fps).set_start(len(frames) * (1 / fps))
        frames.append(frame_clip)

    cap.release()
    return CompositeVideoClip(frames).set_duration(len(frames) / fps)

def extract_event_sequence(prompt):
    # Example simplistic implementation splitting on conjunctions, or use NLP
    events = []
    # Split by keywords indicating event breaks like 'then', 'and suddenly', etc.
    splits = re.split(r",|then|and suddenly|and", prompt.lower())
    for s in splits:
        s = s.strip()
        # Normalize event by replacing spaces with underscores, removing stop words etc.
        event = "_".join(re.findall(r'\w+', s))
        if event:
            events.append(event)
    return events

def generate_video(prompt, output_path):
    print(f"🎬 Prompt: {prompt}")
    keywords = extract_keywords(prompt)
    print(f"🔍 Extracted & Normalized Keywords: {keywords}")

    bg_images, videos_remove_bg, videos_keep_bg = select_media_files(keywords)
    object_paths = select_objects(keywords)
    print(f"🪑 Matched Objects: {[os.path.basename(obj) for obj in object_paths]}")

    if not bg_images:
        print("⚠ No background images found. Will use solid color background.")
        bg_images = []


    if not videos_remove_bg and not videos_keep_bg:
        # Use any video in VIDEO_KEEP_BG_DIR as fallback
        all_vids = [os.path.join(VIDEO_KEEP_BG_DIR, f) for f in os.listdir(VIDEO_KEEP_BG_DIR) if f.lower().endswith('.mp4')]
        if all_vids:
            videos_keep_bg = [all_vids[0]]
        else:
            print("❌ No videos available for fallback.")
            return
    # Helper to get earliest matched keyword index for a file
    def get_earliest_keyword_index(file_path):
        file_tokens = set(tokenize_filename(os.path.basename(file_path)))
        for idx, kw in enumerate(keywords):
            if kw in file_tokens:
                return idx
        return len(keywords)  # if no keyword matches, put it at the end

    # Combine all videos with a tag whether remove_bg or keep_bg
    combined_videos = [(vid, "remove") for vid in videos_remove_bg] + [(vid, "keep") for vid in videos_keep_bg]

    # Sort combined videos by earliest keyword index in prompt keywords
    combined_videos.sort(key=lambda x: get_earliest_keyword_index(x[0]))

    final_clips = []

    for vid, bg_type in combined_videos:
        print(f"🎞 Processing video ({'Remove BG' if bg_type=='remove' else 'Keep BG'}): {vid}")
        action_clip = None
        if bg_type == "remove":
            action_clip = remove_video_bg(vid)
        else:
            action_clip = VideoFileClip(vid).resize(height=720).resize(width=1280)
        action_duration = action_clip.duration

        if bg_images:
            # Try to match background image
            bg_img_path = bg_images[0]  # fallback
            video_tokens = set(tokenize_filename(os.path.basename(vid)))
            for bg in bg_images:
                bg_tokens = set(tokenize_filename(os.path.basename(bg)))
                if video_tokens & bg_tokens:
                    bg_img_path = bg
                    break
                background = (
                        ImageClip(bg_img_path)
                        .set_duration(action_duration)
                        .resize(height=720)
                        .resize(width=1280)
                        )
        else:
            # Use a black screen if no image is available
                background = ColorClip(size=(1280, 720), color=(0, 0, 0)).set_duration(action_duration)


        object_clips = place_objects(object_paths, duration=action_duration)

        start_pos = ("center", 120)
        direction = None
        if "left" in keywords:
            direction = "left"
        elif "right" in keywords:
            direction = "right"

        if ("walk" in keywords or "walking" in keywords) and direction:
            def walk_direction(t):
                start_x = 1280 if direction == "left" else -action_clip.w
                end_x = -action_clip.w if direction == "left" else 1280
                y = 500  # fixed vertical position for walking
                x = start_x + (end_x - start_x) * (t / action_duration)
                return (x, y)
            action_clip = action_clip.set_position(walk_direction)
        elif "travel" in keywords and object_clips:
            target = random.choice([clip.pos(action_duration / 2) for clip in object_clips])
            def moving_position(t):
                x1, y1 = 640 - action_clip.w / 2, 120
                x2, y2 = target
                progress = min(t / action_duration, 1.0)
                return (x1 + (x2 - x1) * progress, y1 + (y2 - y1) * progress)
            action_clip = action_clip.set_position(moving_position)
        else:
            action_clip = action_clip.set_position(start_pos if bg_type == "remove" else "center")

        layers = [background] + object_clips + [action_clip]
        comp = CompositeVideoClip(layers, size=(1280, 720))
        final_clips.append(comp)

    final_video = concatenate_videoclips(final_clips, method="compose")
    final_video.write_videofile(output_path, fps=24)
    print("✅ Video created: final_video.mp4")

def create_video_with_narration(narrative_text, output_file="crime_story.mp4"):
    output_path = os.path.join("uploads", output_file)  # save inside uploads/
    print(f"🎬 Narration: {narrative_text}")
    generate_video(narrative_text, output_path)         # pass full path to video generator
    print(f"✅ Video saved as '{output_path}'\n")
    return output_path
