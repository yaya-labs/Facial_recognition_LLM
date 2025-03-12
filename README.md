# LLM with Facial Recognition

A conversational system that integrates facial recognition capabilities with large language models. The system remembers the people it interacts with and maintains a conversation history for each recognised face.

## How It Works

1. The system utilises InsightFace for real-time face detection and recognition.
2. When a face is detected, it is either identified as a known face or registered as a new one.
3. Conversations with detected faces are stored in memory and recorded to CSV files.
4. The system uses GPT models to generate responses based on:
   - The current conversation,
   - Previous interactions with the detected face(s), and
   - Summaries of who each person is.
5. Face summaries are automatically generated and updated as conversations progress.

## Versions

- **`app.py`** – This version uses OpenAI’s GPT models for all API calls.
- **`app_ASI.py`** – This alternative version uses ASI Mini for all API calls.

## Requirements

- Python 3.7+
- Webcam
- OpenAI API key (for `app.py`) or ASI API key (for `app_ASI.py`)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   
   ```bash
   git clone https://github.com/yaya-labs/Facial_recognition_LLM
   cd Facial_recognition_LLM
   ```

2. Install the dependencies:
   
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API keys:

   - For **`app.py`**, create a `.env` file in the project directory with:
     
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```
     Alternatively, set it as an environment variable.
     
   - For **`app_ASI.py`**, create a `.env` file with:
     
     ```
     ASI_API_KEY=your_asi_api_key_here
     ```
     Alternatively, set it as an environment variable.

## Usage

1. To run the OpenAI version:
   
   ```bash
   python app.py
   ```

2. To run the ASI Mini version:
   
   ```bash
   python app_ASI.py
   ```

3. Commands within the application:
   
   - `/quit` – Exit the application.
   - `/rename` – Rename a detected face.
   - `/faces` – Show all known faces.
   - `/summary` – Show summaries for active faces.
   - `/fix` – Fix CSV encoding issues.

4. Webcam window controls:
   
   - Press `q` to quit.
   - Press `r` to rename a face.

## Training Face Recognition

The system can be trained to recognise specific individuals:

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

2. Start the application – it will automatically process these images and train the facial recognition system.

## Data Storage

The system stores the following data:

- `face_embeddings.pkl` – Data for facial recognition.
- `conversations.csv` – A record of all conversations.
- `face_summaries.csv` – Automatically generated summaries for each person.

## Customisation

To modify the personality or behaviour of the LLM, edit the `prompt` variable in the code.

## Notes

- The system uses **gpt-4o-mini** (for `app.py`) or **asi1-mini** (for `app_ASI.py`) for all API calls.
- Facial recognition performance may depend on good lighting conditions.
- The system automatically manages conversation history to stay within token limits.

## Troubleshooting

- If facial recognition is not working, ensure your webcam is correctly connected.
- For CSV encoding issues, use the `/fix` command.
- If you experience API errors, check your API key and internet connection.

## Credits

- Facial recognition is powered by [InsightFace](https://github.com/deepinsight/insightface).
