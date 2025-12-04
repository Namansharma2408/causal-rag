"""
Transcript manager for accessing full conversation transcripts.
Used for evidence extraction and answer verification.
"""

from typing import List, Dict, Any, Optional
from pymongo import MongoClient

from ..config import Config, logger


class TranscriptManager:
    """Access full transcripts from MongoDB."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._client = MongoClient(self.config.MONGODB_URI)
        self._db = self._client[self.config.TRANSCRIPTS_DB]
        self._collection = self._db[self.config.TRANSCRIPTS_COLLECTION]
    
    def get_transcript(self, transcript_id: str) -> Optional[Dict[str, Any]]:
        """Get full transcript by ID."""
        try:
            doc = self._collection.find_one({"transcript_id": transcript_id})
            return doc
        except Exception as e:
            logger.error(f"Error getting transcript {transcript_id}: {e}")
            return None
    
    def get_transcripts(self, transcript_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple transcripts by IDs."""
        try:
            docs = list(self._collection.find(
                {"transcript_id": {"$in": transcript_ids}}
            ))
            return docs
        except Exception as e:
            logger.error(f"Error getting transcripts: {e}")
            return []
    
    def get_conversation_text(self, transcript_id: str) -> Optional[str]:
        """Get formatted conversation text from transcript."""
        doc = self.get_transcript(transcript_id)
        if not doc:
            return None
        
        turns = doc.get("turns", [])
        if not turns:
            return None
        
        lines = []
        for turn in turns:
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            lines.append(f"{speaker}: {text}")
        
        return "\n".join(lines)
    
    def search_in_transcript(
        self, 
        transcript_id: str, 
        keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """Search for keywords in a transcript's turns."""
        doc = self.get_transcript(transcript_id)
        if not doc:
            return []
        
        matches = []
        turns = doc.get("turns", [])
        
        for i, turn in enumerate(turns):
            text = turn.get("text", "").lower()
            for keyword in keywords:
                if keyword.lower() in text:
                    matches.append({
                        "turn_index": i,
                        "speaker": turn.get("speaker"),
                        "text": turn.get("text"),
                        "keyword": keyword
                    })
                    break
        
        return matches
    
    def close(self):
        """Close MongoDB connection."""
        self._client.close()
