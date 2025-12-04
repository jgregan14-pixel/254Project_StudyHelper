# 254Project_StudyHelper

This project is designed to help a user study through multiple different techniques. The program will alert the user when a camera detects that the user is distracted from their study matierials. It will then activate a buzzer to signal to the user to get back to work. It also features a timer that will track certain studying milestones and then the program will communicate to the user by signaling to an arduino to blink an LED So that the user can put the computer in a seperate location but will be still able to tell when the session is over or a milestone has been hit. It will also use OpenAI's API to send motivational messages that will be given to the user over voice.

**REQUIREMENTS**
Required Hardware: computer with python 3.11.18, webcam, Arduino, LED with all wires and resistors
Required Setup: OpenAI API key, Arduino IDE, Google Teachable model

If you do not have an OpenAI key you can get one from here: https://platform.openai.com/api-keys

**INSTALL**

1. Clone this repository

2. Set up the Environment: Create a file named .env in the main project folder and add your OpenAI API key to it, EXAMPLE: OPENAI_API_KEY=your_api_key_here

3. Install all the required Python libraries using the requirements.txt file, RUN: pip install -r requirements.txt

4. Upload the main.ino sketch to your Arduino and connect it to your computer

**HOW TO USE**

1. Run python main.py

2. You will have the option to upload a PDF for a Q&A section. If you type 'y' you can drag and drop a file into the console

3. Choose your prefered study method

4. Start Working: The model will now watch you and activate the buzzer if you look away

5. Afterwards if you uploaded a PDF you will see the Q&A portion pop up

