import pandas as pd
from enum import Enum, auto
from dataclasses import dataclass
from typing import List
from torch.utils.data import Dataset

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    LLAMA_2 = auto()

@dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]  # ["user", "sys"]
    messages: List[List[str]]  # [[role, message, emotion?, strategy?]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    def process_all(self) -> str:
        """
        Process conversation into format ready for training
        
        Args:
            tokenizer: Tokenizer with <MEM> token
            
        Returns:
            Dict containing input_ids, attention_mask and labels tensors
        """
        input_text = ""
        
        if self.system:
            input_text += f"{self.system} <MEM> "
        
        for msg in self.messages:
            role, text = msg[0], msg[1]
            
            # Add emotion if exists (index 2)
            if len(msg) > 2 and msg[2]:
                text = f"({msg[2]}) {text}"
                
            # Add strategy if exists (index 3) and is system message
            if len(msg) > 3 and msg[3] and role == "sys":
                text = f"({msg[3]}) {text}"
                
            input_text += f"{role}{text} <MEM> "
                    
        return input_text
    
    def process_each(self) -> List[str]:
        """
        Process conversation into format ready for training
        
        Args:
            tokenizer: Tokenizer with <MEM> token
            
        Returns:
            Dict containing input_ids, attention_mask and labels tensors
        """
        input_texts = []
        
        if self.system:
            input_texts.append(f"{self.system} <MEM>")
        
        for msg in self.messages:
            role, text = msg[0], msg[1]
            
            # Add emotion if exists (index 2)
            if len(msg) > 2 and msg[2]:
                text = f"({msg[2]}) {text}"
                
            # Add strategy if exists (index 3) and is system message
            if len(msg) > 3 and msg[3] and role == "sys":
                text = f"({msg[3]}) {text}"
                
            input_texts.append(f"{role}{text} <MEM>")
                    
        return input_texts
    

class ConversationDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Return original text and target text for reconstruction
        return {
            'input_text': text,
            'target_text': text
        }

def csv_to_conversations(path: str) -> List[Conversation]:
    """
    Convert CSV file to Conversation objects. using pandas for handling the data. Sample csv:
    Utterance,Speaker,Emotion,Strategy,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime
    I told you.,Client,sadness,undefined,0,0,1,1,"00:00:54,097","00:00:55,113"
    Told me what?,Therapist,neutral,Open question,0,1,1,1,"00:00:55,260","00:00:56,263"
    That you'd be sorry you ever encouraged me to cry here.,Client,sadness,undefined,0,2,1,1,"00:00:56,651","00:00:59,719"
    I'm not sorry at all.,Therapist,neutral,Communication Skills,0,3,1,1,"00:00:59,951","00:01:01,237"
    "You didn't expect it to be like this, I bet.",Client,sadness,undefined,0,4,1,1,"00:01:03,404","00:01:05,206"
    Like what?,Therapist,neutral,Open question,0,5,1,1,"00:01:05,880","00:01:06,826"
    """
    data = pd.read_csv(path)
    conversations = []

    # Utterances may contains " in the start and end
    data["Utterance"] = data["Utterance"].str.replace('"', '')
    
    for dialogue_id in data["Dialogue_ID"].unique():
        # Get all utterances in the dialogue
        dialogue_data = data[data["Dialogue_ID"] == dialogue_id]
        
        # Convert dialog turns into messages format
        messages = []
        for _, turn in dialogue_data.iterrows():
            # Basic message has role and text
            message = [
                turn["Speaker"],  # role 
                turn["Utterance"],     # text
                turn.get("Emotion"),    # emotion (may be None)
                turn.get("Strategy") if turn["Speaker"] == "Therapist" else None
            ]
            messages.append(message)
            
        # Create Conversation object
        conv = Conversation(
            system="",  # Empty system prompt
            roles=["Client", "Therapist"],
            messages=messages,
            offset=0,
            sep_style=SeparatorStyle.SINGLE,
            sep="###"
        )
        
        conversations.append(conv)
        
    return conversations