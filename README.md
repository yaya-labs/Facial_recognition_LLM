# LLM with Face Recognition

A conversational system  that integrates face recognition capabilities with LLMs. The system remembers people it interacts with and maintains conversation history for each recognised face.

## How It Works

1. The system uses InsightFace for real-time face detection and recognition
2. When a face is detected, it's either identified as a known face or registered as a new one
3. Conversations with detected faces are maintained in memory and stored to CSV files
4. The system uses OpenAI's GPT models to generate responses based on:
   - The current conversation
   - Previous interactions with the detected face(s)
   - Summaries of who each person is
5. Face summaries are automatically generated and updated as conversations progress

## Requirements

- Python 3.7+
- Webcam
- OpenAI API key
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yaya-labs/Facial_recognition_LLM
   cd Facial_recognition_LLM
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Create a `.env` file in the project directory with:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```
   - Alternatively, set it as an environment variable

## Usage

1. Run the application:
   ```bash
   python app.py
   ```

2. Commands:
   - `/quit` - Exit the application
   - `/rename` - Rename a detected face
   - `/faces` - Show all known faces
   - `/summary` - Show summaries for active faces
   - `/fix` - Fix CSV encoding issues

3. Webcam window controls:
   - Press `q` to quit
   - Press `r` to rename a face

## Training Face Recognition

The system can be trained to recognise specific people:

1. Create a folder structure under `training_faces/`:
   ```
   training_faces/
   ├── Person_Name_1/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── Person_Name_2/
   │   ├── image1.jpg
   │   └── ...
   └── ...
   ```

2. Start the application - it will automatically process these images and train the face recognition system

## Data Storage

The system stores:

- `face_embeddings.pkl` - Face recognition data
- `conversations.csv` - Record of all conversations
- `face_summaries.csv` - Generated summaries of each person


## Customisation

To modify the LLM's personality or behaviour, edit the `prompt` variable in the code.

## Notes

- The system uses gpt-4o-mini for all API calls to OpenAI
- Face recognition may require good lighting conditions
- The system automatically manages history to stay within token limits

## Troubleshooting

- If face recognition is not working, ensure your webcam is properly connected
- For CSV encoding issues, use the `/fix` command
- If experiencing OpenAI API errors, check your API key and connection


## Credits
- InsightFace for face recognition: https://github.com/deepinsight/insightface
