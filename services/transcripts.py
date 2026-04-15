from typing import List, Dict, Any, Optional
import importlib

from ..config import Config, logger


class TranscriptManager:
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        pymongo = importlib.import_module("pymongo")
        mongo_client_cls = getattr(pymongo, "MongoClient")
        self._client = mongo_client_cls(self.config.MONGODB_URI)
        self._db = self._client[self.config.TRANSCRIPTS_DB]
        self._collection = self._db[self.config.TRANSCRIPTS_COLLECTION]
    
    def get_transcript(self, transcript_id: str) -> Optional[Dict[str, Any]]:
        try:
            doc = self._collection.find_one({"transcript_id": transcript_id})
            return doc
        except Exception as e:
            logger.error(f"Error getting transcript {transcript_id}: {e}")
            return None
    
    def get_transcripts(self, transcript_ids: List[str]) -> List[Dict[str, Any]]:
        try:
            docs = list(self._collection.find(
                {"transcript_id": {"$in": transcript_ids}}
            ))
            return docs
        except Exception as e:
            logger.error(f"Error getting transcripts: {e}")
            return []
    
    def get_conversation_text(self, transcript_id: str) -> Optional[str]:
        doc = self.get_transcript(transcript_id)
        if not doc:
            return None
        
        turns = doc.get("turns") or doc.get("conversation", [])
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
        doc = self.get_transcript(transcript_id)
        if not doc:
            return []
        
        matches = []
        turns = doc.get("turns") or doc.get("conversation", [])
        
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
        self._client.close()
