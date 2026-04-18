"""
FGA-filtered context retriever for the Vision Agent.

Wraps data retrieval with Auth0 Fine-Grained Authorization (FGA) checks
to ensure the Vision Agent can only access data streams that have been
explicitly authorized by the caregiver. This is the data access layer
that sits BEFORE ContextSnapshot assembly.

Auth0 Features Used: Fine-Grained Authorization (FGA)
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timezone

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from auth0.fga import is_authorized, filter_streams_by_permission

logger = logging.getLogger("vision_agent.fga_retriever")


@dataclass
class DataStream:
    """Represents a data stream that requires FGA authorization."""
    stream_id: str
    stream_type: str  # vision_feed, health_telemetry, location, etc.
    data: Optional[Dict[str, Any]] = None
    authorized: bool = False
    checked_at: Optional[str] = None


@dataclass
class AuthorizedContext:
    """
    FGA-filtered context for the Vision Agent.
    
    Only contains data streams that the agent has been explicitly
    authorized to access via Auth0 FGA relationship tuples.
    """
    agent_id: str
    streams: List[DataStream] = field(default_factory=list)
    denied_streams: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    @property
    def authorized_stream_ids(self) -> List[str]:
        """Return IDs of authorized streams."""
        return [s.stream_id for s in self.streams if s.authorized]
    
    def has_stream(self, stream_id: str) -> bool:
        """Check if a specific stream was authorized."""
        return any(s.stream_id == stream_id and s.authorized for s in self.streams)


class FGAContextRetriever:
    """
    FGA-filtered context retriever for ETMS Vision Agent.
    
    Before the ContextSnapshot is assembled, this retriever checks
    Auth0 FGA to verify the agent has permission to access each
    data stream. Only authorized streams are included in the context.
    
    This implements the principle of least privilege for AI agents:
    agents only see data they've been explicitly granted access to.
    
    Auth0 Feature: Fine-Grained Authorization (Zanzibar-based)
    """
    
    # Available data streams in the ETMS system
    AVAILABLE_STREAMS = [
        "vision_feed",
        "health_telemetry",
        "location",
        "environmental",
        "medication_schedule",
        "emergency_contacts",
    ]
    
    def __init__(self, agent_id: str = "vision_agent"):
        """
        Initialize the FGA context retriever.
        
        Args:
            agent_id: The agent identifier used in FGA relationship tuples.
                     Must match the user in FGA (e.g., 'vision_agent').
        """
        self.agent_id = agent_id
        logger.info(f"FGA Context Retriever initialized for agent: {agent_id}")
    
    def retrieve_authorized_context(
        self,
        requested_streams: Optional[List[str]] = None,
    ) -> AuthorizedContext:
        """
        Retrieve only FGA-authorized data streams for context assembly.
        
        This method checks authorization BEFORE retrieving data, ensuring
        the Vision Agent never sees data it hasn't been granted access to.
        
        Args:
            requested_streams: Specific streams to check. If None, checks all
                             available streams.
        
        Returns:
            AuthorizedContext with only authorized streams populated.
            
        Auth0 Feature: FGA batch check with Zanzibar relationship model
        """
        streams_to_check = requested_streams or self.AVAILABLE_STREAMS
        context = AuthorizedContext(agent_id=self.agent_id)
        
        logger.info(f"Checking FGA authorization for {len(streams_to_check)} streams")
        
        for stream_id in streams_to_check:
            try:
                authorized = is_authorized(
                    user=self.agent_id,
                    relation="viewer",
                    object_type="data_stream",
                    object_id=stream_id,
                )
                
                stream = DataStream(
                    stream_id=stream_id,
                    stream_type=stream_id,
                    authorized=authorized,
                    checked_at=datetime.now(timezone.utc).isoformat(),
                )
                
                if authorized:
                    # Only fetch data for authorized streams
                    stream.data = self._fetch_stream_data(stream_id)
                    context.streams.append(stream)
                    logger.info(f"✓ FGA authorized: {self.agent_id} → {stream_id}")
                else:
                    context.denied_streams.append(stream_id)
                    logger.warning(f"✗ FGA denied: {self.agent_id} → {stream_id}")
                    
            except Exception as e:
                logger.error(f"FGA check failed for {stream_id}: {e}")
                context.denied_streams.append(stream_id)
        
        logger.info(
            f"FGA retrieval complete: {len(context.streams)} authorized, "
            f"{len(context.denied_streams)} denied"
        )
        
        return context
    
    def check_single_stream(self, stream_id: str) -> bool:
        """
        Check FGA authorization for a single data stream.
        
        Args:
            stream_id: The data stream to check access for
            
        Returns:
            True if the agent is authorized to access the stream
            
        Auth0 Feature: FGA single relationship check
        """
        try:
            return is_authorized(
                user=self.agent_id,
                relation="viewer",
                object_type="data_stream",
                object_id=stream_id,
            )
        except Exception as e:
            logger.error(f"FGA check failed for {stream_id}: {e}")
            return False
    
    def get_authorized_stream_list(self) -> List[str]:
        """
        Get list of all streams this agent is authorized to access.
        
        Returns:
            List of authorized stream IDs
            
        Auth0 Feature: FGA batch permission check
        """
        try:
            authorized = filter_streams_by_permission(
                user=self.agent_id,
                relation="viewer",
                object_type="data_stream",
                object_ids=self.AVAILABLE_STREAMS,
            )
            return authorized
        except Exception as e:
            logger.error(f"FGA batch check failed: {e}")
            return []
    
    def _fetch_stream_data(self, stream_id: str) -> Dict[str, Any]:
        """
        Fetch data for an authorized stream.
        
        This is called ONLY after FGA authorization has been verified.
        In production, this would query the actual data sources
        (cameras, sensors, databases).
        
        Args:
            stream_id: The authorized stream to fetch data from
            
        Returns:
            Stream data dictionary
        """
        # Simulated data for each stream type
        stream_data = {
            "vision_feed": {
                "source": "living_room_camera_1",
                "frame_rate": 15,
                "resolution": "640x480",
                "last_detection": "person_standing",
                "pose_confidence": 0.92,
            },
            "health_telemetry": {
                "heart_rate": 72,
                "spo2": 97,
                "temperature": 36.6,
                "activity_level": "low",
                "last_update": datetime.now(timezone.utc).isoformat(),
            },
            "location": {
                "zone": "living_room",
                "confidence": 0.95,
                "in_geofence": True,
                "method": "indoor_positioning",
            },
            "environmental": {
                "temperature": 22.5,
                "humidity": 45,
                "light_level": "normal",
                "noise_level": "quiet",
            },
            "medication_schedule": {
                "next_medication": "14:00",
                "medication_name": "Blood pressure medication",
                "taken_today": True,
            },
            "emergency_contacts": {
                "primary": "caregiver",
                "secondary": "family_member",
                "emergency_services": "112",
            },
        }
        
        return stream_data.get(stream_id, {"stream_id": stream_id, "data": "no_mock_available"})


def create_fga_filtered_context(
    agent_id: str = "vision_agent",
    requested_streams: Optional[List[str]] = None,
) -> AuthorizedContext:
    """
    Convenience function to create an FGA-filtered context.
    
    Use this before assembling the ContextSnapshot in the vision pipeline.
    
    Args:
        agent_id: The agent requesting context
        requested_streams: Optional list of specific streams to check
        
    Returns:
        AuthorizedContext with only permitted data streams
        
    Auth0 Feature: FGA-powered data access control for AI agents
    """
    retriever = FGAContextRetriever(agent_id=agent_id)
    return retriever.retrieve_authorized_context(requested_streams)
