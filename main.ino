int ledPin = 8; 
int buzzerPin = 9;

void setup() {
    pinMode(ledPin, OUTPUT); 
    pinMode(buzzerPin, OUTPUT);
   
    Serial.begin(9600);
    Serial.println("Arduino ready");
}

void loop() {
    // Check if data is available from Python
    if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n');
        command.trim(); // remove any extra spaces or newline chars

        // LIGHT COMMAND: flash LED 5 times
        if (command == "light") {
            for (int i = 0; i < 5; i++) {
                digitalWrite(ledPin, HIGH);
                delay(500);
                digitalWrite(ledPin, LOW);
                delay(500);
            }
        }

        // ALERT COMMAND: play buzzer tones
        else if (command == "alert") { 
            tone(buzzerPin, 440); // A4
            delay(500);
            tone(buzzerPin, 494); // B4
            delay(500);
            tone(buzzerPin, 523); // C5
            delay(500); 
            noTone(buzzerPin);
        }
    }
}
