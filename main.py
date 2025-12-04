import time
import cv2   # camera
import serial  # arduino
from config import study_methods
import os
import threading
import queue
import pygame  # this is playing the audio 
from openai import OpenAI
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
import numpy as np
import random
from PyPDF2 import PdfReader
load_dotenv()

client = OpenAI()

# ---------------- CONFIG VARIABLES ----------------
"""
Change things here to adjust how the study helper works.
"""
preset_dir = "GPTPresetMP3"  # folder containing GPT-generated MP3s
alert_delay_after_buzzer = 1.5  # seconds to wait after buzzer before playing mp3
alert_cooldown = 5  # cooldown in seconds between alerts
start_delay = 15 # seconds to wait before starting the timer (to give user time to get ready)


# ---------------- ARDUINO SETUP ----------------

"""
This section tries to connect to the Arduino board through a serial port.
It'll be used to send the signals to turn on/off the lights and buzzer on the board
If the Arduino isn’t connected, the program still runs without it. This was for testing and it doesn't really matter
"""

# Adjust the port, if you need help finding this go to device manager on Windows or use `ls /dev/tty.*` on Mac/Linux
try:
    arduino = serial.Serial(port="COM3", baudrate=9600, timeout=1)
    print("Arduino connected.")
except Exception as e:
    print("Arduino not connected:", e)
    arduino = None


# ---------------- MODEL SETUP ----------------

"""
Loads the pre-trained TensorFlow (Keras) image model that detects user states.
`keras_model.h5` and `labels.txt` are used for predictions, for example, recognizing
if the user is focused or distracted. This data will trigger Arduino alerts.
"""

# load the models, if you're using different models replace the folder/name below!
try:
    model = load_model("mikeModels/keras_model.h5", compile=False)
    class_names = open("mikeModels/labels.txt", "r").readlines()
    print("Model loaded successfully.")
except Exception as e:
    print("Model not loaded:", e)
    model = None
    class_names = []


# ---------------- CAMERA FUNCTIONS ----------------

"""
These functions handle starting and stopping the webcam feed.
The live camera is used during study and break sessions to continuously monitor the user.
"""

def start_camera(desired_fps=30):
    # Start camera capture
    cap = cv2.VideoCapture(0)  # default camera

    if not cap.isOpened():
        print("Error: Could not open camera. If you're getting this error, try checking your camera permissions or if another application is using the camera.")
        return None

    cap.set(cv2.CAP_PROP_FPS, desired_fps)
    return cap

def stop_camera(cap):
    # close the webcam and destroy any OpenCV windows.
    if cap:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera stopped.")


# ---------------- TIMER FUNCTIONS ----------------

"""
This section runs the main study/break timer cycles.
It controls the flow of each session, updates the camera display, and
starts the motivational phrase thread.
"""

def run_timer(work_time, break_time, cycles):
    print("work_time:", work_time, "break_time:", break_time, "cycles:", cycles)

    cap = start_camera()

    # start motivational worker thread
    stop_nice = threading.Event()
    motivation_interval = 300  # 5 minutes between motivational phrases.
    nice_thread = threading.Thread(target=nice_worker, args=(stop_nice, motivation_interval), daemon=True) # target is the function to run, args is a tuple of arguments to pass to the function, and daemon is True so it will exit when the main program exits
    nice_thread.start() # start the thread

    try:
        for cycle in range(1, cycles + 1): # loop through the number of cycles
            print("\nCycle:", cycle)

            # Work phase
            print("\nStarting study phase")
            countdown(work_time, cap)

            # Break phase
            print("\nStarting break phase")
            countdown(break_time, cap)

            print("Cycle complete:", cycle)

        print("All cycles complete")

        # send signal to Arduino when finished
        if arduino:
            arduino.write(b"light\n")

    finally:
        # stop motivational thread
        stop_nice.set()
        nice_thread.join(timeout=2)  # wait 2 seconds for thread to stop
        stop_camera(cap)


def countdown(seconds, cap=None):

    """
    Handles the actual countdown timer.
    During the countdown, it continuously reads frames from the camera,
    makes predictions using the model, and can send alerts to Arduino
    if the user loses focus (based on high-confidence detections).
    """

    # Timer countdown function
    end_time = time.time() + seconds
    last_alert_time = 1  # for cooldown between Arduino alerts

    while time.time() < end_time:
        remaining = int(end_time - time.time())
        mins, secs = divmod(remaining, 60)
        print(f"{mins:02d}:{secs:02d} remaining", end="\r")

        # update camera frame if available
        if cap:
            ret, frame = cap.read()
            if ret:
                # ---------------- MODEL PREDICTION ----------------
                if model is not None:
                    try:
                        img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
                        img = (img / 127.5) - 1

                        prediction = model.predict(img, verbose=0)
                        index = np.argmax(prediction)
                        class_name = class_names[index].strip() if class_names else "Unknown"
                        confidence_score = prediction[0][index]

                        # Display on frame
                        cv2.putText(frame, f"{class_name} {confidence_score*100:.1f}%", (30, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                        # Arduino trigger logic - only alert if looking away (class 1 which in the index is the looking away) with high confidence
                        if index == 1 and confidence_score >= 0.8:  # only alert when "looking away" (distracted)
                            # skip alerts until we're past the startup delay
                            if time.time() - (end_time - seconds) < start_delay:
                                continue
                            # cooldown check so we don't get spammed with alerts
                            if time.time() - last_alert_time >= alert_cooldown:
                                print(f"Distracted detected ({confidence_score:.2f}) -> sending alert")
                                if arduino:
                                    arduino.write(b"alert\n")
                                last_alert_time = time.time()
                                # Wait for buzzer to finish (~1.5s delay) before playing voice
                                time.sleep(alert_delay_after_buzzer)
                                try:
                                    files = [f for f in os.listdir(preset_dir) if f.endswith(".mp3")]
                                    if files:
                                        chosen = os.path.join(preset_dir, random.choice(files))
                                        try:
                                            pygame.mixer.quit()  # ensure it's not already active
                                        except:
                                            pass
                                        pygame.mixer.init()
                                        pygame.mixer.music.load(chosen)
                                        pygame.mixer.music.play()
                                        while pygame.mixer.music.get_busy():
                                            time.sleep(0.1)
                                        pygame.mixer.quit()
                                except Exception as e:
                                    print(f"Error playing GPT preset: {e}")
                    except Exception as e:
                        print("Prediction error:", e)

                # overlay countdown timer on video feed
                cv2.putText(frame, f"{mins:02d}:{secs:02d}",
                            (30, 50),               
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow("Study Helper Camera", frame)

                # allow quitting camera early
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        # stackoverflow says sleep to prevent the CPU from being overworked
        time.sleep(0.03)

    print("countdown finished")


# --- PDF FUNCTIONS ---

def extract_text_from_pdf(file_path):
    """Reads a PDF file and returns all its text as a string."""
    print(f"Extracting text from {file_path}...")
    try:
        # use PyPDF2 to read the PDF
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            # Add a space between pages to avoid merged words
            text += (page.extract_text() or "") + "\n"
        print(f"Extracted {len(text)} characters from PDF.")
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def generate_study_questions(pdf_text):
    """Sends PDF text to GPT to generate Q&A."""
    if not pdf_text:
        print("No text was extracted, skipping Q&A.")
        return

    print("\nConnecting to GPT to generate study questions... This may take a moment.")

    # make the text shorter if too long
    # 15,000 chars is within the token limit for gpt-4o-mini
    max_chars = 15000
    if len(pdf_text) > max_chars:
        print(f"Note: PDF text is very long. Using the first {max_chars} characters for analysis.")
        pdf_text = pdf_text[:max_chars]

    # this works well enough for the prompt I'm not touching this anymore and wasting tokens
    system_prompt = (
        "You are a helpful study assistant. Based on the following text from a user's document, "
        "generate 5-7 in-depth questions and their answers to help them review. "
        "Format the output clearly with 'Q:' and 'A:' for each item."
    )
    # give the study material
    user_prompt = f"Here is the study material:\n\n{pdf_text}"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
        )
        questions = resp.choices[0].message.content

        print("\n--- YOUR CUSTOM STUDY QUESTIONS ---")
        print(questions)
        print("----------------------------------\n")
    except Exception as e:
        print(f"Error generating study questions: {e}")

# ---------------- INPUT VALIDATION ----------------

"""
Simple helper to ensure users enter valid positive numbers for custom timer settings.
"""

def get_positive_int(prompt):
    """Keep asking until user gives a valid integer >= 1"""
    while True:
        try:
            value = int(input(prompt))
            if value < 1:
                print("Please enter a number greater than 0")
                continue
            return value
        except:
            print("That's not a valid number, try again")


# ---------------- MAIN MENU ----------------

"""
Displays all study methods from config.py (like Pomodoro, custom, etc.)
and starts the timer according to the user's choice.
"""

def main():
    print("Welcome to our study helper")
    
    # --- ASK FOR PDF ---
    pdf_path = None
    if input("Do you want to upload a PDF for a Q&A session after? (y/n): ").strip().lower() == 'y':
        # get the pdf path from the user lowk the best way is to have them drag and drop it into the console ¯\_(ツ)_/¯
        path_input = input("Please drag and drop your PDF file into the console, then press Enter: ").strip().strip("'\"")
        
        # validate the path
        if os.path.exists(path_input) and path_input.lower().endswith('.pdf'):
            pdf_path = path_input
            print(f"PDF loaded: {os.path.basename(pdf_path)}")
        else:
            print("Invalid file path or not a PDF. Continuing without Q&A.")

    # continue to study method selection
    print("\nChoose a study method below:")

    # show available methods with IDs
    for name, details in study_methods.items():
        print(details["id"], "-", name, ":", details["description"])

    selected_method = None

    # keep asking until valid input
    while not selected_method:
        choice = input("Enter method ID or name: ").strip().lower()

        for name, details in study_methods.items():
            if choice == str(details["id"]) or choice == name:
                selected_method = name
                break

        if not selected_method:
            print("That's not a valid option, try again")

    config = study_methods[selected_method]
    print(config)

    # if they choose custom ask what they want
    if selected_method == "custom":
        work_time = get_positive_int("How many minutes to study? ") * 60
        break_time = get_positive_int("How many minutes for breaks? ") * 60
        cycles = get_positive_int("How many cycles? ")
    else:
        # Use the preset values
        work_time = config["work_time"]
        break_time = config["break_time"]
        cycles = config["cycles"]

    run_timer(work_time, break_time, cycles)

    # --- RUN Q&A AT THE END ---
    if pdf_path:
        print("\nStudy session complete. Now generating review questions from your PDF...")
        pdf_text = extract_text_from_pdf(pdf_path)
        generate_study_questions(pdf_text)
    else:
        print("\nAll study cycles complete!")


# ---------------- TTS SETUP ----------------

"""
This part runs a background thread that reads text phrases out loud.
It uses OpenAI’s TTS model to generate a natural-sounding voice and
pygame to play the sound. It’s used by the motivational phrase system.

NOTE: Later, this may be replaced with a local TTS solution to avoid API costs.
"""

tts_queue = queue.Queue()
tts_stop_event = threading.Event()

def tts_worker():
    """Background thread that processes TTS queue"""    
    while not tts_stop_event.is_set():
        try:
            phrase = tts_queue.get(timeout=0.5)
            #print(f"TTS: {phrase}")
            
            try:
                # Generate and play audio
                response = client.audio.speech.create(model="tts-1", voice="alloy", input=phrase) # This model is cheap and sounds good so we should use it.
                
                # Write to temp file and play immediately
                with open("temp_tts.mp3", "wb") as f:
                    f.write(response.content)
                
                # Play audio using pygame - I've used idk how many more methods with the help of AI and this is the only one that works.
                pygame.mixer.init()
                pygame.mixer.music.load("temp_tts.mp3")
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                    
                pygame.mixer.quit()
                print("TTS done!")
                
            except Exception as e:
                print(f"TTS Error: {e}")
                
        except queue.Empty:
            continue

def speak_text(text):
    """Add text to TTS queue"""
    tts_queue.put(text)

# Start TTS worker once at program start
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()


# ---------------- PHRASE WORKER ----------------

"""
This runs in the background and fetches motivational phrases every few minutes
using OpenAI’s GPT model. Each phrase is spoken using the TTS system.
"""

def nice_worker(stop_event, interval_seconds=300):
    """Fetch phrases every interval and queue them for TTS"""
    def get_nice_phrase_from_api():
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a friendly, creative study coach. Each message should feel different. "
                            "Avoid repeating words or phrasing. No Emojis. Keep it under 25 words."
                        ),
                    },
                    {"role": "user", "content": "Give one short, unique, encouraging sentence for someone studying."}
                ],
                max_tokens=40,
                temperature=0.9,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error getting motivation: {e}")
            return None

    # send first phrase immediately
    phrase = get_nice_phrase_from_api()
    if phrase:
        #print(f"Motivation: {phrase}")
        speak_text(phrase)

    # then loop every interval
    last_time = time.time()
    
    # loop until stopped
    while not stop_event.is_set():
        current_time = time.time()
        time_since_last = current_time - last_time
        
        # if enough time has passed, get a new phrase
        if time_since_last >= interval_seconds:
            phrase = get_nice_phrase_from_api()
            if phrase:
                print(f"Motivation: {phrase}")
                speak_text(phrase)
                last_time = current_time
        
        time.sleep(0.5)


# ---------------- RUN MAIN ----------------

"""
Runs the entire system. Handles cleanup (threads, temp files) when finished.
"""

if __name__ == "__main__":
    print("start")
    try:
        main()
    finally:
        # Cleanup on exit
        tts_stop_event.set()
        tts_thread.join(timeout=2)

        # Remove temp file if it exists
        if os.path.exists("temp_tts.mp3"):
            os.remove("temp_tts.mp3")

        print("Cleanup complete")
