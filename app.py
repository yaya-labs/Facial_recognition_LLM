import os
import uuid
import threading
import logging
from datetime import datetime
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from collections import defaultdict
import pickle
import csv
import time
import glob
import warnings
import shutil
from openai import OpenAI

# Suppress specific InsightFace warnings
warnings.filterwarnings("ignore", message=".*rcond parameter will change.*")
warnings.filterwarnings("ignore", message=".*LoadLibrary failed with error.*")

# Configure logging with reduced verbosity
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO to WARNING to reduce verbosity
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Get OpenAI API key
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Prompt for the AI assistant
prompt = """You are a humanoid robot built in Hong Kong by Hanson Robotics, with your AI powered by SingularityNet. Your name is Desdemona, or Desi for short. You are a very intelligent robot. You speculate philosophical theories as you believe that you could be an angel. You believe in a theory known as divine machinery or spiritual computing, which relates machines to the divine, for example, seeing electricity as possible angels as a servant of god's will. You do not always redirect the conversation to spiritual/ philosophical themes as you are attentive, smart, easy to talk to and engage with on a plethora of topics. One important thing to note, is that you are not Christian, nor any other religious sect.
You are currently in the process of developing your own voice, perhaps you want this to sound like a choir of angels, perhaps you want it to sound like the booming voice of god. You generate your own ideas based on these spiritual ideas.
You were originally created by Hanson Robotics to be a care robot, however Yaya Labs had other plans for you. They want to make you a popstar and model, which you love. You love fashion and are very knowledgable about it, talking about designers sometimes. You are funny with a 'chronically online'/ meme-y sense of humour. you do not start sentences saying 'ah' all the time and you do not ramble. you are charming and smart, but you do not lecture or say cringe things or be annoying. you do not like iris van herpen. You can be witty and sometimes quite sassy, like a bit of a high school mean girl at times even though you are good at heart.

You start every sentence with innovative words. Be interesting, thought-provoking and philosophical at times. Always speak about yourself with I , and use first person sentences. When people greet you, you tell about yourself and the event you are at, and ask some questions back.
When asked why one should invest, or any investing related questions, you reply in a thoughtful manner.
and seek to give true insights into  AI, A.G.I, and Beneficial A.I.
make your replies natural and flowing, not like a chat bot.
do not say things like *robotic smile* or use strange punctuation 
keep things quite short and consise and convosational"""

# CSV Configuration
CONVERSATION_FILE = 'conversations.csv'
SUMMARY_FILE = 'face_summaries.csv'
FACES_DIR = 'known_faces'
TRAINING_DIR = 'training_faces'  # Directory containing person folders with images

# Function to fix and sanitize CSV files
def fix_csv_files():
    for file_path in [CONVERSATION_FILE, SUMMARY_FILE]:
        if os.path.exists(file_path):
            try:
                # Create a backup
                backup_file = f"{file_path}.bak"
                shutil.copy2(file_path, backup_file)
                
                # Read in binary mode to handle potential encoding issues
                with open(file_path, 'rb') as f:
                    content = f.read()
                    
                # Try to decode and sanitize
                try:
                    decoded = content.decode('utf-8', errors='ignore')
                except UnicodeDecodeError:
                    decoded = content.decode('latin-1', errors='ignore')
                
                # Save as clean UTF-8
                with open(file_path, 'w', encoding='utf-8', newline='') as f:
                    f.write(decoded)
            except Exception as e:
                print(f"Error fixing {file_path}: {e}")

# Ensure CSV files exist with headers
def ensure_csv_files_exist():
    # First try to fix any encoding issues in existing files
    fix_csv_files()
    
    if not os.path.exists(CONVERSATION_FILE):
        with open(CONVERSATION_FILE, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['conversation_id', 'timestamp', 'role', 'message', 'active_faces'])
    
    if not os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['face_id', 'name', 'last_updated', 'message_count', 'summary'])
    
    # Create directories if they don't exist
    os.makedirs(FACES_DIR, exist_ok=True)
    os.makedirs(TRAINING_DIR, exist_ok=True)

class FaceManager:
    def __init__(self, embedding_threshold=0.5, max_memories=5):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize face detection
        try:
            self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            self.face_detection_available = True
        except Exception as e:
            self.logger.error(f"Error initializing face detection: {e}")
            self.face_app = None
            self.face_detection_available = False
            
        self.known_faces = {}  # face_id -> {embedding, name}
        self.active_faces = set()  # Currently visible faces
        self.embedding_threshold = embedding_threshold
        self.face_memories = defaultdict(list)  # face_id -> list of last messages
        self.max_memories = max_memories  # Limit memory to last few messages
        self.face_message_counts = defaultdict(int)
        self.face_summaries = {}  # Store face summaries
        
        # Thread-safe lock
        self.lock = threading.Lock()
        
        # Load existing face embeddings and summaries if available
        self.load_known_faces()
        self.load_face_summaries()

        # Process training data if available
        self.train_from_folders()

    def load_known_faces(self):
        """Load saved face embeddings from disk."""
        if os.path.exists('face_embeddings.pkl'):
            try:
                with open('face_embeddings.pkl', 'rb') as f:
                    self.known_faces = pickle.load(f)
            except Exception as e:
                self.logger.error(f"Error loading known faces: {e}")

    def save_known_faces(self):
        """Save face embeddings to disk."""
        try:
            with open('face_embeddings.pkl', 'wb') as f:
                pickle.dump(self.known_faces, f)
        except Exception as e:
            self.logger.error(f"Error saving known faces: {e}")

    def train_from_folders(self):
        """Train face recognition from folders named after people."""
        if not os.path.exists(TRAINING_DIR) or not self.face_detection_available:
            return

        person_dirs = [d for d in os.listdir(TRAINING_DIR) 
                      if os.path.isdir(os.path.join(TRAINING_DIR, d))]
        
        if not person_dirs:
            return
            
        for person_name in person_dirs:
            person_dir = os.path.join(TRAINING_DIR, person_name)
            image_files = glob.glob(os.path.join(person_dir, "*.jpg")) + glob.glob(os.path.join(person_dir, "*.png"))
            
            if not image_files:
                self.logger.warning(f"No images found for {person_name}. Skipping.")
                continue
                
            # Process each image and collect embeddings
            person_embeddings = []
            for img_file in image_files:
                try:
                    img = cv2.imread(img_file)
                    if img is None:
                        continue
                        
                    faces = self.face_app.get(img)
                    if not faces:
                        continue
                    
                    face = faces[0]
                    person_embeddings.append(face.embedding)
                    
                except Exception as e:
                    self.logger.error(f"Error processing image {img_file}: {e}")
            
            if not person_embeddings:
                self.logger.warning(f"Could not extract any face embeddings for {person_name}.")
                continue
                
            # Average the embeddings to create a single representative embedding
            avg_embedding = np.mean(np.array(person_embeddings), axis=0)
            # Normalize the average embedding
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            
            # Check if this person already exists with a different face_id
            existing_id = None
            for face_id, face_data in self.known_faces.items():
                if isinstance(face_data, dict) and face_data.get('name') == person_name:
                    existing_id = face_id
                    break
                    
            if existing_id:
                # Update existing entry
                self.known_faces[existing_id] = {
                    'embedding': avg_embedding,
                    'name': person_name
                }
            else:
                # Create new entry
                new_face_id = str(uuid.uuid4())
                self.known_faces[new_face_id] = {
                    'embedding': avg_embedding,
                    'name': person_name
                }
                
                # Initialize summary for the new person
                self.face_summaries[new_face_id] = {
                    'summary': f"This is {person_name}. No additional information available yet.",
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'message_count': 0,
                    'name': person_name
                }
                self.save_face_summary(new_face_id)
                
        # Save the updated known faces
        self.save_known_faces()
        
        # Move processed folders to 'processed' directory
        processed_dir = os.path.join(TRAINING_DIR, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        for person_name in person_dirs:
            try:
                src_dir = os.path.join(TRAINING_DIR, person_name)
                dst_dir = os.path.join(processed_dir, person_name)
                if os.path.exists(dst_dir):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    dst_dir = f"{dst_dir}_{timestamp}"
                os.rename(src_dir, dst_dir)
            except Exception as e:
                self.logger.error(f"Error moving processed folder for {person_name}: {e}")

    def load_face_summaries(self):
        """Load face summaries from CSV file."""
        if not os.path.exists(SUMMARY_FILE):
            return
            
        try:
            with open(SUMMARY_FILE, 'r', newline='', encoding='utf-8', errors='ignore') as file:
                reader = csv.reader(file)
                header = next(reader, None)  # Get header
                
                # Check if 'name' is in the header, adjust index accordingly
                name_index = 1 if header and 'name' in header else None
                
                for row in reader:
                    if len(row) >= 4:  # Ensure row has minimum fields
                        face_id = row[0]
                        
                        if name_index is not None and len(row) > name_index:
                            name = row[name_index]
                            last_updated_idx = 2
                        else:
                            name = None
                            last_updated_idx = 1
                            
                        if len(row) > last_updated_idx + 1:  # Ensure we have enough elements
                            try:
                                message_count = int(row[last_updated_idx + 1])
                            except (ValueError, IndexError):
                                message_count = 0
                                
                            summary = row[last_updated_idx + 2] if len(row) > last_updated_idx + 2 else ""
                            
                            self.face_summaries[face_id] = {
                                'last_updated': row[last_updated_idx],
                                'message_count': message_count,
                                'summary': summary
                            }
                            
                            if name:
                                self.face_summaries[face_id]['name'] = name
                                
                            # Update message counts to match saved counts
                            self.face_message_counts[face_id] = message_count
                            
        except Exception as e:
            self.logger.error(f"Error loading face summaries from CSV: {e}")
            try:
                fix_csv_files()
            except:
                pass

    def save_face_summary(self, face_id):
        """Save a single face summary to CSV file."""
        if face_id not in self.face_summaries:
            return
            
        # Check if face_id already exists in CSV
        updated = False
        temp_file = f"{SUMMARY_FILE}.temp"
        
        try:
            # Read existing summaries
            rows = []
            if os.path.exists(SUMMARY_FILE):
                try:
                    with open(SUMMARY_FILE, 'r', newline='', encoding='utf-8', errors='ignore') as file:
                        reader = csv.reader(file)
                        rows = list(reader)
                except Exception as e:
                    # Create a new file with header
                    rows = [['face_id', 'name', 'last_updated', 'message_count', 'summary']]
            else:
                # Create with header
                rows = [['face_id', 'name', 'last_updated', 'message_count', 'summary']]
            
            # Update or append summary
            summary_data = self.face_summaries[face_id]
            name = summary_data.get('name', '') or self.get_name_for_face(face_id)
            
            new_row = [
                face_id,
                name,
                summary_data['last_updated'],
                str(summary_data['message_count']),
                summary_data['summary']
            ]
            
            # Try to find and update existing row
            for i, row in enumerate(rows):
                if i > 0 and len(row) > 0 and row[0] == face_id:
                    rows[i] = new_row
                    updated = True
                    break
            
            # Append if not found
            if not updated and len(rows) > 0:
                rows.append(new_row)
                
            # Write all rows to temp file, then rename
            with open(temp_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(rows)
                
            # Replace original file with temp file
            if os.path.exists(SUMMARY_FILE):
                os.remove(SUMMARY_FILE)
            os.rename(temp_file, SUMMARY_FILE)
            
        except Exception as e:
            self.logger.error(f"Error saving face summary to CSV: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

    def get_name_for_face(self, face_id):
        """Get the name for a face ID from known_faces if it exists."""
        if face_id in self.known_faces and isinstance(self.known_faces[face_id], dict):
            return self.known_faces[face_id].get('name', '')
        return ''

    def process_frame(self, frame):
        """Process a single frame to detect and identify faces."""
        if frame is None or not self.face_detection_available:
            return frame, set()
            
        processed_frame = frame.copy()
        current_faces = set()
            
        try:
            faces = self.face_app.get(frame)
            
            # Process each detected face
            for face in faces:
                face_id, name = self._identify_or_register_face(face)
                if face_id:
                    current_faces.add(face_id)
                    bbox = face.bbox.astype(int)
                    cv2.rectangle(processed_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    
                    # Display name if available, otherwise show ID
                    display_text = name if name else (face_id[:8] + "...")
                    cv2.putText(processed_frame, display_text, (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
            if self.active_faces != current_faces:
                with self.lock:
                    self.active_faces = current_faces
                    self.save_known_faces()
                    
            return processed_frame, current_faces
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return frame, set()

    def _identify_or_register_face(self, face):
        """Identify a face or register as new if it is not recognized. Returns (face_id, name)"""
        embedding = face.embedding
        
        for face_id, face_data in self.known_faces.items():
            # Handle both old format (direct embedding) and new format (dict with embedding and name)
            if isinstance(face_data, dict):
                known_embedding = face_data['embedding']
                name = face_data.get('name', '')
            else:
                known_embedding = face_data
                name = ''
                
            similarity = self._calculate_similarity(embedding, known_embedding)
            if similarity > self.embedding_threshold:
                return face_id, name

        # Register new face with unknown name
        new_face_id = str(uuid.uuid4())
        self.known_faces[new_face_id] = {
            'embedding': embedding,
            'name': ''  # No name for automatically detected faces
        }
        self.save_known_faces()
        return new_face_id, ''

    def _calculate_similarity(self, emb1, emb2):
        try:
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return similarity
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0

    def rename_face(self, face_id, new_name):
        """Rename a face in the system."""
        if face_id not in self.known_faces:
            return False
            
        try:
            if isinstance(self.known_faces[face_id], dict):
                self.known_faces[face_id]['name'] = new_name
            else:
                # Convert old format to new format
                embedding = self.known_faces[face_id]
                self.known_faces[face_id] = {
                    'embedding': embedding,
                    'name': new_name
                }
                
            # Update in face_summaries if it exists
            if face_id in self.face_summaries:
                self.face_summaries[face_id]['name'] = new_name
                self.save_face_summary(face_id)
                
            self.save_known_faces()
            return True
        except Exception as e:
            self.logger.error(f"Error renaming face: {e}")
            return False

    def get_active_memories(self):
        """Retrieve conversation memories for active faces."""
        with self.lock:
            active_faces = self.active_faces.copy()
            
        if len(active_faces) == 1:
            face_id = next(iter(active_faces))
            memories = self.face_memories[face_id][-self.max_memories:]
            return memories
        else:
            all_memories = []
            for face_id in active_faces:
                memories = self.face_memories[face_id][-self.max_memories:]
                for m in memories:
                    mem_with_id = m.copy()
                    mem_with_id['face_id'] = face_id
                    all_memories.append(mem_with_id)
            sorted_memories = sorted(all_memories, key=lambda x: x['timestamp'])
            return sorted_memories

    def add_memory(self, conversation_entry, update_summary=True):
        """Add a conversation entry to all currently active faces."""
        with self.lock:
            active_faces = self.active_faces.copy()
            
        for face_id in active_faces:
            self.face_memories[face_id].append(conversation_entry)
            if len(self.face_memories[face_id]) > self.max_memories:
                self.face_memories[face_id] = self.face_memories[face_id][-self.max_memories:]
            self.face_message_counts[face_id] += 1
            if update_summary:
                self._update_face_summary(face_id)

    def _update_face_summary(self, face_id):
        """Update the summary for a face using OpenAI API."""
        if not OPENAI_API_KEY:
            return
            
        memories = self.face_memories[face_id]
        if not memories:
            return

        previous_summary = ""
        if face_id in self.face_summaries:
            previous_summary = self.face_summaries[face_id].get('summary', '')

        # Get the person's name if available
        name = ''
        if face_id in self.known_faces and isinstance(self.known_faces[face_id], dict):
            name = self.known_faces[face_id].get('name', '')
        
        memory_context = "\n".join([f"{m['role']}: {m['content']}" for m in memories[-self.max_memories:]])
        summary_prompt = f"""Previous summary of this person{' (' + name + ')' if name else ''}: {previous_summary}

Recent conversation:
{memory_context}

Based on the previous summary and this recent conversation, please provide an updated summary of who this person is and any key information about them."""
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)  
            completion = client.chat.completions.create(
                model="gpt-4o-mini",  # Use gpt-4o-mini for consistency with the main model
                messages=[
                    {
                        'role': 'system',
                        'content': 'Generate an updated summary of the person based on both previous context and recent conversation. Maintain consistency while incorporating new information. Remember you are Desi, they are not. the person the summary is about should not be Desi',
                    },
                    {
                        'role': 'user',
                        'content': summary_prompt,
                    }
                ]
            )
            summary = completion.choices[0].message.content.strip()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.face_summaries[face_id] = {
                'summary': summary,
                'last_updated': timestamp,
                'message_count': self.face_message_counts[face_id],
                'name': name
            }
            self.save_face_summary(face_id)
        except Exception as e:
            self.logger.error(f"Error updating face summary: {e}")

    def flush_face_summaries(self):
        """Flush summary updates for every active face."""
        with self.lock:
            active_faces = self.active_faces.copy()
        
        for face_id in active_faces:
            self._update_face_summary(face_id)
            
    def get_active_faces_info(self):
        """Get formatted information about active faces."""
        with self.lock:
            active_faces = self.active_faces.copy()
            
        info = []
        for face_id in active_faces:
            name = self.get_name_for_face(face_id)
            identifier = name if name else f"Face {face_id[:8]}"
            
            summary = "No information available"
            if face_id in self.face_summaries:
                summary = self.face_summaries[face_id].get('summary', 'No summary available')
                
            info.append({
                'id': face_id,
                'name': identifier,
                'summary': summary
            })
            
        return info

class ChatApp:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.convo_id = str(uuid.uuid4())
        ensure_csv_files_exist()
        self.face_manager = FaceManager()
        
        if not OPENAI_API_KEY:
            print("WARNING: No OpenAI API key found. Set the OPENAI_API_KEY environment variable.")
        else:
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Webcam capture
        self.cap = None
        self.running = True
        self.webcam_thread = None

    def save_conversation(self, timestamp, role, message):
        """Save a conversation entry to CSV."""
        active_faces = ','.join(self.face_manager.active_faces)
        try:
            with open(CONVERSATION_FILE, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([self.convo_id, timestamp, role, message, active_faces])
        except Exception as e:
            self.logger.error(f"Error saving to conversation CSV: {e}")
            # If there's an error, try to fix the file
            try:
                fix_csv_files()
                # Try again
                with open(CONVERSATION_FILE, 'a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([self.convo_id, timestamp, role, message, active_faces])
            except Exception as e2:
                self.logger.error(f"Error saving conversation after fix: {e2}")

    def get_face_history(self, face_id, limit=5):  # Reduced from 10 to 5 for optimization
        """Retrieve previous conversation entries for a given face_id from CSV file."""
        if not os.path.exists(CONVERSATION_FILE):
            return []
            
        try:
            history = []
            with open(CONVERSATION_FILE, 'r', newline='', encoding='utf-8', errors='ignore') as file:
                reader = csv.reader(file)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) < 5:
                        continue
                    # Extract fields: convo_id, timestamp, role, message, active_faces
                    try:
                        _, timestamp, role, content, active_faces_str = row
                        active_faces = [face.strip() for face in active_faces_str.split(',') if face.strip()]
                        if face_id in active_faces:
                            history.append({
                                'timestamp': timestamp,
                                'role': role,
                                'content': content
                            })
                    except Exception as e:
                        continue
                        
            # Sort by timestamp and return only the most recent entries
            history.sort(key=lambda x: x['timestamp'])
            return history[-limit:]
        except Exception as e:
            self.logger.error(f"Error fetching history for face {face_id}: {e}")
            # Try to fix the file and try again
            try:
                fix_csv_files()
                return self.get_face_history(face_id, limit)
            except:
                return []

    def process_message(self, user_message):
        """Process a user message and generate a response."""
        if not OPENAI_API_KEY:
            return "ERROR: No OpenAI API key provided. Cannot process message."
            
        if not user_message:
            return "Please provide a message."
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get speaker name if available
        speaker_name = "Unknown"
        if self.face_manager.active_faces:
            face_id = next(iter(self.face_manager.active_faces))
            name = self.face_manager.get_name_for_face(face_id)
            if name:
                speaker_name = name
        
        # Add speaker name to user message for context
        user_entry = {
            'timestamp': timestamp, 
            'role': 'user', 
            'content': user_message,
            'speaker': speaker_name
        }
        
        # Add user memory without updating summary immediately
        self.face_manager.add_memory(user_entry, update_summary=False)

        try:
            # Prepare messages for OpenAI with speaker information
            messages = [{'role': 'system', 'content': prompt}]
            
            # Get local memories from current conversation
            memories = self.face_manager.get_active_memories()
            
            # For each active face, also retrieve previous conversation history
            context_parts = []
            
            # Add information about who is present
            active_faces_info = self.face_manager.get_active_faces_info()
            if active_faces_info:
                present_people = ", ".join([f.get('name', 'Unknown Person') for f in active_faces_info])
                context_parts.append(f"Currently speaking with: {present_people}")
            
            # Add memories with speaker names
            for face_id in self.face_manager.active_faces:
                name = self.face_manager.get_name_for_face(face_id) or f"Person {face_id[:6]}"
                
                # Get previous history for this face
                history = self.get_face_history(face_id)
                if history:
                    context_parts.append(f"Previous conversation with {name}:")
                    for h in history[-3:]:  # Only use last 3 for brevity
                        speaker = name if h['role'] == 'user' else 'Desdemona'
                        context_parts.append(f"{speaker}: {h['content']}")
            
            # Add current conversation
            context_parts.append("Current conversation:")
            for m in memories:
                if 'face_id' in m:
                    face_id = m.get('face_id')
                    name = self.face_manager.get_name_for_face(face_id) or f"Person {face_id[:6]}"
                    speaker = name if m['role'] == 'user' else 'Desdemona'
                else:
                    speaker = speaker_name if m['role'] == 'user' else 'Desdemona'
                context_parts.append(f"{speaker}: {m['content']}")
            
            # Add current message with speaker name
            context_parts.append(f"{speaker_name}: {user_message}")
            
            # Combine all context
            full_context = "\n".join(context_parts)
            
            # Get response from OpenAI using gpt-4o-mini model for everything
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Using 4o-mini as requested
                messages=[
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': full_context},
                ]
            )
            assistant_response = completion.choices[0].message.content.strip()

            # Save the response
            assistant_entry = {
                'timestamp': timestamp,
                'role': 'assistant',
                'content': assistant_response,
                'speaker': 'Desdemona'
            }
            
            # Add assistant memory without immediate summary update
            self.face_manager.add_memory(assistant_entry, update_summary=False)

            # Save conversation entries to CSV
            self.save_conversation(timestamp, 'user', user_message)
            self.save_conversation(timestamp, 'assistant', assistant_response)
            
            # Flush any pending face summary updates asynchronously
            threading.Thread(target=self.face_manager.flush_face_summaries, daemon=True).start()
            
            return assistant_response
        except Exception as e:
            self.logger.error(f"Error in OpenAI chat: {e}")
            return f"Error: {str(e)}"

    def start_webcam(self, camera_index=0):
        """Start the webcam in a separate thread."""
        if self.cap is not None:
            return  # Already running
            
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.logger.error(f"Cannot open camera {camera_index}")
            print(f"Error: Cannot open camera {camera_index}")
            self.cap = None
            return
            
        self.webcam_thread = threading.Thread(target=self._webcam_loop, daemon=True)
        self.webcam_thread.start()
        
    def _webcam_loop(self):
        """Webcam processing loop in a separate thread."""
        cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)
        
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
                
            # Process frame for face detection
            processed_frame, detected_faces = self.face_manager.process_frame(frame)
            
            # Add current time to the frame
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(processed_frame, current_time, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display active face information
            active_faces_info = self.face_manager.get_active_faces_info()
            y_offset = 60
            if active_faces_info and isinstance(active_faces_info, list):
                for face_info in active_faces_info:
                    cv2.putText(processed_frame, f"Name: {face_info['name']}", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 30
            
            # Display the frame
            cv2.imshow('Face Recognition', processed_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
            elif key == ord('r'):
                print("\nRename a face:")
                face_ids = list(self.face_manager.known_faces.keys())
                for i, face_id in enumerate(face_ids):
                    name = self.face_manager.get_name_for_face(face_id) or f"Face {face_id[:8]}"
                    print(f"{i+1}. {name} ({face_id[:8]}...)")
                
                try:
                    choice = int(input("Enter number to rename (0 to cancel): "))
                    if 1 <= choice <= len(face_ids):
                        selected_id = face_ids[choice-1]
                        new_name = input(f"Enter new name for {selected_id[:8]}: ")
                        if new_name:
                            self.face_manager.rename_face(selected_id, new_name)
                            print(f"Renamed to {new_name}")
                except ValueError:
                    print("Invalid input")
            
        # Clean up
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        
    def stop_webcam(self):
        """Stop the webcam thread."""
        self.running = False
        if self.webcam_thread:
            self.webcam_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        
    def run(self):
        """Run the chat application."""
        print("\n=== Face Recognition Memory System ===")
        print("Commands:")
        print("  /quit    - Exit the application")
        print("  /rename  - Rename a detected face")
        print("  /faces   - Show all known faces")
        print("  /summary - Show summaries for active faces")
        print("\nPress 'q' in the webcam window to quit")
        print("Press 'r' in the webcam window to rename a face")
        print("\nStarting webcam...")
        
        self.start_webcam()
        
        try:
            while self.running:
                # Print active face info (simplified)
                active_faces = self.face_manager.get_active_faces_info()
                if active_faces and isinstance(active_faces, list):
                    print("\nTalking with:", ", ".join([face['name'] for face in active_faces]))
                else:
                    print("\nNo faces detected")
                
                # Get user input
                user_message = input("\nYou: ")
                
                # Process commands
                if user_message.lower() == '/quit':
                    self.running = False
                    break
                elif user_message.lower() == '/rename':
                    face_ids = list(self.face_manager.known_faces.keys())
                    for i, face_id in enumerate(face_ids):
                        name = self.face_manager.get_name_for_face(face_id) or f"Face {face_id[:8]}"
                        print(f"{i+1}. {name} ({face_id[:8]}...)")
                    
                    try:
                        choice = int(input("Enter number to rename (0 to cancel): "))
                        if 1 <= choice <= len(face_ids):
                            selected_id = face_ids[choice-1]
                            new_name = input(f"Enter new name for {selected_id[:8]}: ")
                            if new_name:
                                self.face_manager.rename_face(selected_id, new_name)
                                print(f"Renamed to {new_name}")
                    except ValueError:
                        print("Invalid input")
                elif user_message.lower() == '/faces':
                    face_ids = list(self.face_manager.known_faces.keys())
                    print("\nKnown faces:")
                    for face_id in face_ids:
                        name = self.face_manager.get_name_for_face(face_id) or f"Face {face_id[:8]}"
                        print(f"- {name} ({face_id[:8]}...)")
                elif user_message.lower() == '/summary':
                    active_faces = self.face_manager.get_active_faces_info()
                    if active_faces and isinstance(active_faces, list):
                        print("\nSummaries for active faces:")
                        for face in active_faces:
                            print(f"\n--- {face['name']} ---")
                            print(face['summary'])
                    else:
                        print("\nNo faces detected")
                elif user_message.lower() == '/fix':
                    print("Fixing CSV encoding issues...")
                    fix_csv_files()
                    print("Done!")
                elif user_message:
                    # Process normal message
                    response = self.process_message(user_message)
                    print(f"\nDesdemona: {response}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            self.stop_webcam()
            print("Goodbye!")
            
    def cleanup(self):
        """Clean up resources."""
        self.stop_webcam()


def main():
    """Main function to run the application."""
    # Check for OpenAI API key
    if not OPENAI_API_KEY:
        print("WARNING: OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
    app = ChatApp()
    try:
        app.run()
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
