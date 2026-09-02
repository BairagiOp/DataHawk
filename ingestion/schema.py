"""
DataHawk Social Media Intelligence Platform
Social Media Data Schema

Defines the unified data model for social media posts across all platforms.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class Platform(str, Enum):
    """Supported platforms"""
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    WEB = "web"
    CSV = "csv"
    JSON = "json"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class MediaType(str, Enum):
    """Media types"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    LINK = "link"
    MIXED = "mixed"


class Language(str, Enum):
    """Supported languages"""
    ENGLISH = "en"
    HINDI = "hi"
    HINGLISH = "hi-en"  # Mixed Hindi-English
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
class Engagement:
    """Engagement metrics for a post"""
    likes: int = 0
    comments: int = 0
    shares: int = 0
    views: int = 0
    retweets: int = 0  # Twitter-specific
    upvotes: int = 0   # Reddit-specific
    downvotes: int = 0 # Reddit-specific

    @property
    def total_engagement(self) -> int:
        """Total engagement score (simple sum)"""
        return (self.likes + self.comments + self.shares +
                self.views + self.retweets + self.upvotes)

    @property
    def engagement_score(self) -> float:
        """Weighted engagement score"""
        return (
            self.likes * 1.0 +
            self.comments * 2.0 +  # Comments show deeper engagement
            self.shares * 3.0 +    # Shares have highest weight
            self.retweets * 2.5 +
            self.upvotes * 1.0 +
            (self.views * 0.01 if self.views > 0 else 0)  # Views matter less
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "likes": self.likes,
            "comments": self.comments,
            "shares": self.shares,
            "views": self.views,
            "retweets": self.retweets,
            "upvotes": self.upvotes,
            "downvotes": self.downvotes,
        }


@dataclass
class SocialMediaPost:
    """
    Unified social media post schema.

    This is the core data structure used throughout the DataHawk pipeline.
    All data sources (CSV, JSON, web scraping, APIs) are converted to this format.

    Required Fields:
        - post_id: Unique identifier
        - platform: Source platform
        - timestamp: When the post was created
        - text: Post content

    Optional Fields:
        - All other fields with defaults

    Derived Fields (populated by pipeline):
        - clean_text: Cleaned version of text
        - sentiment: positive/negative/neutral/mixed
        - emotion: joy/anger/sadness/fear/surprise/disgust/neutral
        - entities: Named entities extracted
        - keywords: Keywords extracted
        - topic_id: Assigned topic cluster
        - trend_score: Computed trend score
    """

    # ── Required Core Fields ────────────────────────────────────
    post_id: str
    platform: Platform
    timestamp: datetime
    text: str

    # ── Optional Core Fields ────────────────────────────────────
    author_id_hash: Optional[str] = None  # Anonymized author ID
    language: Language = Language.UNKNOWN
    url: Optional[str] = None

    # ── Content Attributes ──────────────────────────────────────
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    media_type: MediaType = MediaType.TEXT
    media_urls: List[str] = field(default_factory=list)

    # ── Engagement Metrics ──────────────────────────────────────
    engagement: Engagement = field(default_factory=Engagement)

    # ── Metadata ────────────────────────────────────────────────
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Derived Fields (Populated by Pipeline) ──────────────────
    clean_text: Optional[str] = None
    tokens: List[str] = field(default_factory=list)

    # NLP Analysis Results
    sentiment: Optional[str] = None  # positive/negative/neutral/mixed
    sentiment_score: Optional[float] = None  # [-1, 1]
    sentiment_confidence: Optional[float] = None

    emotion: Optional[str] = None  # joy/anger/sadness/fear/surprise/disgust/neutral
    emotion_confidence: Optional[float] = None

    entities: List[Dict[str, Any]] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    # Topic & Trend Assignment
    topic_id: Optional[int] = None
    topic_label: Optional[str] = None
    topic_confidence: Optional[float] = None

    trend_score: Optional[float] = None
    trend_category: Optional[str] = None  # EMERGING/RISING/STABLE/DECLINING/VIRAL

    # Quality Flags
    is_duplicate: bool = False
    is_spam: bool = False
    spam_score: Optional[float] = None

    # Processing Status
    processed: bool = False
    processing_errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate and normalize data after initialization"""
        # Ensure platform is enum
        if isinstance(self.platform, str):
            try:
                self.platform = Platform(self.platform.lower())
            except ValueError:
                self.platform = Platform.UNKNOWN

        # Ensure language is enum
        if isinstance(self.language, str):
            try:
                self.language = Language(self.language.lower())
            except ValueError:
                self.language = Language.UNKNOWN

        # Ensure media_type is enum
        if isinstance(self.media_type, str):
            try:
                self.media_type = MediaType(self.media_type.lower())
            except ValueError:
                self.media_type = MediaType.TEXT

        # Ensure timestamp is datetime
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))

        # Initialize clean_text if not set
        if self.clean_text is None:
            self.clean_text = self.text

        # Ensure metadata has essential fields
        if 'collection_date' not in self.metadata:
            self.metadata['collection_date'] = datetime.now().isoformat()

    def to_dict(self, include_derived: bool = True) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = {
            # Core fields
            "post_id": self.post_id,
            "platform": self.platform.value,
            "timestamp": self.timestamp.isoformat(),
            "text": self.text,
            "author_id_hash": self.author_id_hash,
            "language": self.language.value,
            "url": self.url,

            # Content
            "hashtags": self.hashtags,
            "mentions": self.mentions,
            "media_type": self.media_type.value,
            "media_urls": self.media_urls,

            # Engagement
            "engagement": self.engagement.to_dict(),

            # Metadata
            "metadata": self.metadata,
        }

        if include_derived:
            data.update({
                "clean_text": self.clean_text,
                "tokens": self.tokens,
                "sentiment": self.sentiment,
                "sentiment_score": self.sentiment_score,
                "sentiment_confidence": self.sentiment_confidence,
                "emotion": self.emotion,
                "emotion_confidence": self.emotion_confidence,
                "entities": self.entities,
                "keywords": self.keywords,
                "topic_id": self.topic_id,
                "topic_label": self.topic_label,
                "topic_confidence": self.topic_confidence,
                "trend_score": self.trend_score,
                "trend_category": self.trend_category,
                "is_duplicate": self.is_duplicate,
                "is_spam": self.is_spam,
                "spam_score": self.spam_score,
                "processed": self.processed,
                "processing_errors": self.processing_errors,
            })

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SocialMediaPost":
        """Create from dictionary"""
        # Extract engagement
        engagement_data = data.get("engagement", {})
        engagement = Engagement(**engagement_data) if engagement_data else Engagement()

        # Create post
        post = cls(
            post_id=data["post_id"],
            platform=data.get("platform", Platform.UNKNOWN),
            timestamp=data["timestamp"],
            text=data["text"],
            author_id_hash=data.get("author_id_hash"),
            language=data.get("language", Language.UNKNOWN),
            url=data.get("url"),
            hashtags=data.get("hashtags", []),
            mentions=data.get("mentions", []),
            media_type=data.get("media_type", MediaType.TEXT),
            media_urls=data.get("media_urls", []),
            engagement=engagement,
            metadata=data.get("metadata", {}),
        )

        # Set derived fields if present
        if "clean_text" in data:
            post.clean_text = data["clean_text"]
        if "tokens" in data:
            post.tokens = data["tokens"]
        if "sentiment" in data:
            post.sentiment = data["sentiment"]
        if "sentiment_score" in data:
            post.sentiment_score = data["sentiment_score"]
        if "sentiment_confidence" in data:
            post.sentiment_confidence = data["sentiment_confidence"]
        if "emotion" in data:
            post.emotion = data["emotion"]
        if "emotion_confidence" in data:
            post.emotion_confidence = data["emotion_confidence"]
        if "entities" in data:
            post.entities = data["entities"]
        if "keywords" in data:
            post.keywords = data["keywords"]
        if "topic_id" in data:
            post.topic_id = data["topic_id"]
        if "topic_label" in data:
            post.topic_label = data["topic_label"]
        if "topic_confidence" in data:
            post.topic_confidence = data["topic_confidence"]
        if "trend_score" in data:
            post.trend_score = data["trend_score"]
        if "trend_category" in data:
            post.trend_category = data["trend_category"]
        if "is_duplicate" in data:
            post.is_duplicate = data["is_duplicate"]
        if "is_spam" in data:
            post.is_spam = data["is_spam"]
        if "spam_score" in data:
            post.spam_score = data["spam_score"]
        if "processed" in data:
            post.processed = data["processed"]
        if "processing_errors" in data:
            post.processing_errors = data["processing_errors"]

        return post

    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate post data.

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # Required fields
        if not self.post_id:
            errors.append("post_id is required")

        if not self.text or len(self.text.strip()) < 3:
            errors.append("text must be at least 3 characters")

        if not isinstance(self.timestamp, datetime):
            errors.append("timestamp must be a datetime object")

        # Validate engagement values
        if self.engagement.likes < 0:
            errors.append("engagement.likes cannot be negative")
        if self.engagement.comments < 0:
            errors.append("engagement.comments cannot be negative")
        if self.engagement.shares < 0:
            errors.append("engagement.shares cannot be negative")
        if self.engagement.views < 0:
            errors.append("engagement.views cannot be negative")

        return (len(errors) == 0, errors)

    def __repr__(self) -> str:
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return (
            f"SocialMediaPost(id={self.post_id}, platform={self.platform.value}, "
            f"timestamp={self.timestamp.isoformat()}, text='{text_preview}')"
        )


@dataclass
class PostBatch:
    """
    Batch of posts with metadata.

    Used for batch processing and experiment tracking.
    """
    posts: List[SocialMediaPost]
    batch_id: str
    source: str
    collection_date: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.posts)

    @property
    def platforms(self) -> List[Platform]:
        """Unique platforms in this batch"""
        return list(set(p.platform for p in self.posts))

    @property
    def languages(self) -> List[Language]:
        """Unique languages in this batch"""
        return list(set(p.language for p in self.posts))

    @property
    def date_range(self) -> tuple[datetime, datetime]:
        """Date range (min, max) of posts"""
        if not self.posts:
            return (self.collection_date, self.collection_date)
        timestamps = [p.timestamp for p in self.posts]
        return (min(timestamps), max(timestamps))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "source": self.source,
            "collection_date": self.collection_date.isoformat(),
            "size": self.size,
            "platforms": [p.value for p in self.platforms],
            "languages": [l.value for l in self.languages],
            "date_range": [d.isoformat() for d in self.date_range()],
            "metadata": self.metadata,
            "posts": [p.to_dict() for p in self.posts],
        }


# ── Validation Utilities ────────────────────────────────────────


def validate_post_dict(data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate a post dictionary before creating SocialMediaPost.

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    # Required fields
    if "post_id" not in data:
        errors.append("post_id is required")

    if "text" not in data:
        errors.append("text is required")
    elif not data["text"] or len(str(data["text"]).strip()) < 3:
        errors.append("text must be at least 3 characters")

    if "timestamp" not in data:
        errors.append("timestamp is required")

    if "platform" not in data:
        errors.append("platform is required")

    return (len(errors) == 0, errors)


def create_post_from_raw(data: Dict[str, Any], platform: Platform = Platform.CSV) -> SocialMediaPost:
    """
    Create SocialMediaPost from raw data with flexible field mapping.

    Handles common variations in field names:
    - timestamp: created_at, date, time, published_at
    - text: content, body, message, tweet, post
    - author_id_hash: author, user_id, username
    - etc.
    """
    import hashlib
    from datetime import datetime

    # Generate post_id if not present
    post_id = data.get("post_id") or data.get("id") or data.get("tweet_id")
    if not post_id:
        # Generate from text hash
        text = str(data.get("text", ""))
        post_id = hashlib.md5(text.encode()).hexdigest()[:16]

    # Extract timestamp with fallback options
    timestamp = None
    for key in ["timestamp", "created_at", "date", "time", "published_at", "datetime"]:
        if key in data and data[key]:
            try:
                if isinstance(data[key], datetime):
                    timestamp = data[key]
                else:
                    timestamp = datetime.fromisoformat(str(data[key]).replace('Z', '+00:00'))
                break
            except:
                continue

    if not timestamp:
        timestamp = datetime.now()

    # Extract text with fallback options
    text = ""
    for key in ["text", "content", "body", "message", "tweet", "post", "description"]:
        if key in data and data[key]:
            text = str(data[key])
            break

    # Extract author
    author = data.get("author_id_hash") or data.get("author") or data.get("user_id") or data.get("username")
    if author:
        # Hash the author ID for privacy
        author_hash = hashlib.sha256(str(author).encode()).hexdigest()[:16]
    else:
        author_hash = None

    # Extract engagement
    engagement = Engagement(
        likes=int(data.get("likes", 0) or 0),
        comments=int(data.get("comments", 0) or 0),
        shares=int(data.get("shares", 0) or 0),
        views=int(data.get("views", 0) or 0),
        retweets=int(data.get("retweets", 0) or 0),
        upvotes=int(data.get("upvotes", 0) or 0),
        downvotes=int(data.get("downvotes", 0) or 0),
    )

    # Extract hashtags
    hashtags = data.get("hashtags", [])
    if isinstance(hashtags, str):
        hashtags = [h.strip() for h in hashtags.split(",")]

    # Extract mentions
    mentions = data.get("mentions", [])
    if isinstance(mentions, str):
        mentions = [m.strip() for m in mentions.split(",")]

    return SocialMediaPost(
        post_id=post_id,
        platform=platform,
        timestamp=timestamp,
        text=text,
        author_id_hash=author_hash,
        language=Language(data.get("language", "unknown").lower()) if data.get("language") else Language.UNKNOWN,
        url=data.get("url"),
        hashtags=hashtags,
        mentions=mentions,
        media_type=MediaType(data.get("media_type", "text").lower()) if data.get("media_type") else MediaType.TEXT,
        engagement=engagement,
        metadata={
            "source": platform.value,
            "raw_data": data,  # Keep original for traceability
        }
    )
