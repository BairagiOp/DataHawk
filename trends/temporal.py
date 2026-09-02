"""
Temporal Analysis for Trend Detection

Implements time-series aggregation and feature engineering for topics.
Tracks volume, engagement, and growth over time windows.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TemporalMetrics:
    """
    Temporal metrics for a topic at a specific time point.
    """
    topic_id: int
    topic_label: str
    timestamp: datetime
    time_window: str  # '1H', '1D', '1W'

    # Volume metrics
    volume: int
    unique_authors: int

    # Engagement metrics
    total_likes: int
    total_comments: int
    total_shares: int
    total_views: int
    total_engagement: int

    # Growth metrics (compared to previous period)
    volume_growth: Optional[float] = None
    engagement_growth: Optional[float] = None

    # Historical context
    baseline_volume: Optional[float] = None
    days_active: Optional[int] = None


class TemporalAnalyzer:
    """
    Analyze topic activity over time.

    Aggregates posts by time windows and computes temporal features
    needed for trend detection.
    """

    def __init__(self, time_window: str = '1D'):
        """
        Args:
            time_window: Pandas frequency string ('1H', '1D', '1W')
        """
        self.time_window = time_window

    def aggregate_by_time(self,
                         posts: List[Dict],
                         topic_assignments: Dict[str, int],
                         topic_labels: Dict[int, str]) -> pd.DataFrame:
        """
        Aggregate posts by time window and topic.

        Args:
            posts: List of post dicts with required fields
            topic_assignments: Dict mapping post_id -> topic_id
            topic_labels: Dict mapping topic_id -> topic_label

        Returns:
            DataFrame with columns:
            - timestamp
            - topic_id
            - topic_label
            - volume (count of posts)
            - unique_authors
            - total_likes, total_comments, total_shares, total_views
            - total_engagement
        """
        # Convert to DataFrame
        df = pd.DataFrame(posts)

        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Add topic assignment
        df['topic_id'] = df['post_id'].map(topic_assignments)

        # Filter out unassigned posts
        df = df[df['topic_id'].notna()]

        # Add topic labels
        df['topic_label'] = df['topic_id'].map(topic_labels)

        # Extract engagement fields (handle missing values)
        df['likes'] = df.get('likes', 0).fillna(0)
        df['comments'] = df.get('comments', 0).fillna(0)
        df['shares'] = df.get('shares', 0).fillna(0)
        df['views'] = df.get('views', 0).fillna(0)

        # Aggregate by time window and topic
        aggregated = df.groupby([
            pd.Grouper(key='timestamp', freq=self.time_window),
            'topic_id',
            'topic_label'
        ]).agg({
            'post_id': 'count',  # volume
            'author_id_hash': 'nunique',  # unique authors
            'likes': 'sum',
            'comments': 'sum',
            'shares': 'sum',
            'views': 'sum'
        }).reset_index()

        # Rename columns
        aggregated.columns = [
            'timestamp', 'topic_id', 'topic_label',
            'volume', 'unique_authors',
            'total_likes', 'total_comments', 'total_shares', 'total_views'
        ]

        # Compute total engagement
        aggregated['total_engagement'] = (
            aggregated['total_likes'] +
            aggregated['total_comments'] +
            aggregated['total_shares'] +
            aggregated['total_views'] * 0.01  # Views weighted less
        )

        return aggregated

    def compute_growth_rates(self, aggregated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute growth rates for each topic.

        Adds columns:
        - volume_growth: % change in volume from previous period
        - engagement_growth: % change in engagement from previous period
        - volume_growth_absolute: absolute change in volume

        Args:
            aggregated_df: Output from aggregate_by_time

        Returns:
            DataFrame with growth columns added
        """
        df = aggregated_df.copy()

        # Sort by topic and timestamp
        df = df.sort_values(['topic_id', 'timestamp'])

        # Compute growth rates by topic
        df['volume_prev'] = df.groupby('topic_id')['volume'].shift(1)
        df['engagement_prev'] = df.groupby('topic_id')['total_engagement'].shift(1)

        # Percentage growth
        df['volume_growth'] = (
            (df['volume'] - df['volume_prev']) / df['volume_prev']
        ).replace([np.inf, -np.inf], np.nan).fillna(0) * 100

        df['engagement_growth'] = (
            (df['total_engagement'] - df['engagement_prev']) / df['engagement_prev']
        ).replace([np.inf, -np.inf], np.nan).fillna(0) * 100

        # Absolute growth
        df['volume_growth_absolute'] = df['volume'] - df['volume_prev']

        # Drop temporary columns
        df = df.drop(['volume_prev', 'engagement_prev'], axis=1)

        return df

    def compute_baseline_metrics(self,
                                 aggregated_df: pd.DataFrame,
                                 baseline_periods: int = 3) -> pd.DataFrame:
        """
        Compute baseline metrics for each topic.

        Adds columns:
        - baseline_volume: average volume over baseline periods
        - days_active: number of periods with activity

        Args:
            aggregated_df: Aggregated data
            baseline_periods: Number of periods for baseline

        Returns:
            DataFrame with baseline columns
        """
        df = aggregated_df.copy()

        # Sort by topic and timestamp
        df = df.sort_values(['topic_id', 'timestamp'])

        # Compute rolling baseline (looking back)
        df['baseline_volume'] = (
            df.groupby('topic_id')['volume']
            .rolling(window=baseline_periods, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        # Compute days active (cumulative count of non-zero periods)
        df['days_active'] = (
            df.groupby('topic_id')['volume']
            .apply(lambda x: (x > 0).cumsum())
            .reset_index(0, drop=True)
        )

        return df

    def compute_velocity_acceleration(self, aggregated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute velocity and acceleration metrics.

        Velocity: Average growth rate over last 3 periods
        Acceleration: Change in growth rate (second derivative)

        Args:
            aggregated_df: Aggregated data with growth rates

        Returns:
            DataFrame with velocity and acceleration columns
        """
        df = aggregated_df.copy()

        # Sort by topic and timestamp
        df = df.sort_values(['topic_id', 'timestamp'])

        # Compute velocity (average of last 3 growth rates)
        df['velocity'] = (
            df.groupby('topic_id')['volume_growth']
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        # Compute acceleration (change in growth rate)
        df['growth_prev'] = df.groupby('topic_id')['volume_growth'].shift(1)
        df['acceleration'] = df['volume_growth'] - df['growth_prev']
        df = df.drop('growth_prev', axis=1)

        return df

    def get_topic_timeline(self,
                          aggregated_df: pd.DataFrame,
                          topic_id: int) -> pd.DataFrame:
        """
        Get complete timeline for a specific topic.

        Args:
            aggregated_df: Aggregated data
            topic_id: Topic ID to extract

        Returns:
            DataFrame for single topic with all metrics
        """
        return aggregated_df[aggregated_df['topic_id'] == topic_id].copy()

    def get_latest_metrics(self, aggregated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get latest metrics for all topics.

        Args:
            aggregated_df: Aggregated data

        Returns:
            DataFrame with one row per topic (latest timestamp)
        """
        # Get latest timestamp for each topic
        latest = aggregated_df.sort_values('timestamp').groupby('topic_id').tail(1)
        return latest.reset_index(drop=True)


def aggregate_by_time(posts: List[Dict],
                     topic_assignments: Dict[str, int],
                     topic_labels: Dict[int, str],
                     time_window: str = '1D') -> pd.DataFrame:
    """
    Convenience function for temporal aggregation.

    Args:
        posts: List of post dicts
        topic_assignments: post_id -> topic_id mapping
        topic_labels: topic_id -> label mapping
        time_window: Time window for aggregation

    Returns:
        Aggregated DataFrame with temporal metrics
    """
    analyzer = TemporalAnalyzer(time_window=time_window)
    aggregated = analyzer.aggregate_by_time(posts, topic_assignments, topic_labels)
    aggregated = analyzer.compute_growth_rates(aggregated)
    aggregated = analyzer.compute_baseline_metrics(aggregated)
    aggregated = analyzer.compute_velocity_acceleration(aggregated)
    return aggregated


# Testing
if __name__ == "__main__":
    print("Temporal Analysis Tests:\n")

    # Create sample data
    from datetime import datetime, timedelta

    sample_posts = []
    base_date = datetime(2026, 8, 1)

    # Topic 1: Growing trend
    for day in range(10):
        volume = 10 + day * 5  # Linear growth
        for i in range(volume):
            sample_posts.append({
                'post_id': f'post_{day}_{i}',
                'timestamp': base_date + timedelta(days=day),
                'author_id_hash': f'author_{i % 20}',
                'text': f'Post about topic 1 day {day}',
                'likes': np.random.randint(0, 50),
                'comments': np.random.randint(0, 20),
                'shares': np.random.randint(0, 10),
                'views': np.random.randint(100, 500)
            })

    # Topic 2: Stable trend
    for day in range(10):
        volume = 50  # Constant volume
        for i in range(volume):
            sample_posts.append({
                'post_id': f'post2_{day}_{i}',
                'timestamp': base_date + timedelta(days=day),
                'author_id_hash': f'author_{i % 30}',
                'text': f'Post about topic 2 day {day}',
                'likes': np.random.randint(0, 30),
                'comments': np.random.randint(0, 10),
                'shares': np.random.randint(0, 5),
                'views': np.random.randint(50, 300)
            })

    # Create topic assignments
    topic_assignments = {}
    for post in sample_posts:
        if 'post2_' in post['post_id']:
            topic_assignments[post['post_id']] = 2
        else:
            topic_assignments[post['post_id']] = 1

    topic_labels = {
        1: "Growing Topic",
        2: "Stable Topic"
    }

    # Run temporal analysis
    analyzer = TemporalAnalyzer(time_window='1D')
    aggregated = analyzer.aggregate_by_time(sample_posts, topic_assignments, topic_labels)
    aggregated = analyzer.compute_growth_rates(aggregated)
    aggregated = analyzer.compute_baseline_metrics(aggregated)
    aggregated = analyzer.compute_velocity_acceleration(aggregated)

    print("Aggregated Temporal Metrics:")
    print(aggregated[['timestamp', 'topic_label', 'volume', 'volume_growth', 'velocity']].head(15))
    print()

    # Get latest metrics
    latest = analyzer.get_latest_metrics(aggregated)
    print("\nLatest Metrics for Each Topic:")
    print(latest[['topic_label', 'volume', 'volume_growth', 'baseline_volume', 'velocity']])
