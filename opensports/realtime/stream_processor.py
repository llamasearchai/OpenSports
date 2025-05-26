"""
Real-time Stream Processor

High-performance stream processing for live sports data using Kafka, Redis,
and async processing for real-time analytics and insights.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, AsyncGenerator
import numpy as np
import pandas as pd
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import redis.asyncio as redis
from opensports.core.config import settings
from opensports.core.logging import get_logger
from opensports.core.database import get_database
from opensports.core.cache import cache_async_result

logger = get_logger(__name__)


class StreamProcessor:
    """
    High-performance real-time stream processor for sports data.
    
    Features:
    - Kafka integration for message streaming
    - Redis for real-time caching and pub/sub
    - Async processing for high throughput
    - Event-driven architecture
    - Scalable processing pipelines
    """
    
    def __init__(self):
        self.kafka_producer = None
        self.kafka_consumer = None
        self.redis_client = None
        self.db = get_database()
        self.processing_pipelines = {}
        self.event_handlers = {}
        self.is_running = False
        
    async def initialize(self):
        """Initialize streaming connections and resources."""
        logger.info("Initializing stream processor")
        
        # Initialize Kafka producer
        try:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                batch_size=16384,
                linger_ms=10,
                buffer_memory=33554432
            )
            logger.info("Kafka producer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            
        # Initialize Redis client
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
    
    async def start_processing(self, topics: List[str] = None):
        """Start the stream processing pipeline."""
        if not topics:
            topics = ['game_events', 'player_stats', 'team_updates', 'betting_odds']
        
        logger.info(f"Starting stream processing for topics: {topics}")
        self.is_running = True
        
        # Start consumer tasks for each topic
        tasks = []
        for topic in topics:
            task = asyncio.create_task(self._consume_topic(topic))
            tasks.append(task)
        
        # Start processing pipeline tasks
        pipeline_task = asyncio.create_task(self._run_processing_pipelines())
        tasks.append(pipeline_task)
        
        # Start health monitoring
        health_task = asyncio.create_task(self._monitor_health())
        tasks.append(health_task)
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Stream processing error: {e}")
        finally:
            self.is_running = False
    
    async def stop_processing(self):
        """Stop the stream processing pipeline."""
        logger.info("Stopping stream processing")
        self.is_running = False
        
        if self.kafka_producer:
            self.kafka_producer.close()
        
        if self.redis_client:
            await self.redis_client.close()
    
    async def publish_event(
        self,
        topic: str,
        event_data: Dict[str, Any],
        key: Optional[str] = None
    ) -> bool:
        """
        Publish an event to a Kafka topic.
        
        Args:
            topic: Kafka topic name
            event_data: Event data to publish
            key: Optional partition key
            
        Returns:
            Success status
        """
        try:
            # Add metadata
            enriched_event = {
                **event_data,
                'timestamp': datetime.now().isoformat(),
                'source': 'opensports',
                'version': '1.0'
            }
            
            # Publish to Kafka
            future = self.kafka_producer.send(topic, enriched_event, key=key)
            record_metadata = future.get(timeout=10)
            
            # Cache in Redis for immediate access
            cache_key = f"event:{topic}:{key or 'latest'}"
            await self.redis_client.setex(
                cache_key,
                300,  # 5 minute TTL
                json.dumps(enriched_event)
            )
            
            logger.debug(f"Published event to {topic}: {record_metadata}")
            return True
            
        except KafkaError as e:
            logger.error(f"Failed to publish event to {topic}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error publishing event: {e}")
            return False
    
    async def _consume_topic(self, topic: str):
        """Consume messages from a Kafka topic."""
        logger.info(f"Starting consumer for topic: {topic}")
        
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                group_id=f'opensports-{topic}',
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset='latest',
                enable_auto_commit=True,
                auto_commit_interval_ms=1000
            )
            
            while self.is_running:
                message_batch = consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        await self._process_message(topic, message)
                        
        except Exception as e:
            logger.error(f"Error consuming from topic {topic}: {e}")
        finally:
            if 'consumer' in locals():
                consumer.close()
    
    async def _process_message(self, topic: str, message):
        """Process a single message from Kafka."""
        try:
            event_data = message.value
            event_key = message.key
            
            # Add processing metadata
            event_data['processed_at'] = datetime.now().isoformat()
            event_data['topic'] = topic
            event_data['partition'] = message.partition
            event_data['offset'] = message.offset
            
            # Route to appropriate handler
            if topic in self.event_handlers:
                await self.event_handlers[topic](event_data)
            else:
                await self._default_event_handler(topic, event_data)
            
            # Update processing metrics
            await self._update_processing_metrics(topic)
            
        except Exception as e:
            logger.error(f"Error processing message from {topic}: {e}")
    
    async def _default_event_handler(self, topic: str, event_data: Dict[str, Any]):
        """Default handler for events without specific handlers."""
        logger.debug(f"Processing event from {topic}: {event_data.get('event_type', 'unknown')}")
        
        # Store in Redis for real-time access
        cache_key = f"latest:{topic}"
        await self.redis_client.setex(
            cache_key,
            3600,  # 1 hour TTL
            json.dumps(event_data)
        )
        
        # Store in database for persistence
        await self._store_event_in_db(topic, event_data)
    
    async def register_event_handler(
        self,
        topic: str,
        handler: Callable[[Dict[str, Any]], None]
    ):
        """Register a custom event handler for a topic."""
        self.event_handlers[topic] = handler
        logger.info(f"Registered event handler for topic: {topic}")
    
    async def create_processing_pipeline(
        self,
        name: str,
        input_topics: List[str],
        processor_func: Callable,
        output_topic: Optional[str] = None
    ):
        """
        Create a custom processing pipeline.
        
        Args:
            name: Pipeline name
            input_topics: List of input topics to consume
            processor_func: Processing function
            output_topic: Optional output topic for results
        """
        pipeline = {
            'name': name,
            'input_topics': input_topics,
            'processor_func': processor_func,
            'output_topic': output_topic,
            'created_at': datetime.now().isoformat(),
            'processed_count': 0,
            'error_count': 0
        }
        
        self.processing_pipelines[name] = pipeline
        logger.info(f"Created processing pipeline: {name}")
    
    async def _run_processing_pipelines(self):
        """Run all registered processing pipelines."""
        while self.is_running:
            try:
                for pipeline_name, pipeline in self.processing_pipelines.items():
                    await self._execute_pipeline(pipeline)
                
                await asyncio.sleep(1)  # Process every second
                
            except Exception as e:
                logger.error(f"Error in processing pipelines: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _execute_pipeline(self, pipeline: Dict[str, Any]):
        """Execute a single processing pipeline."""
        try:
            # Get recent events from input topics
            input_data = {}
            for topic in pipeline['input_topics']:
                cache_key = f"latest:{topic}"
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    input_data[topic] = json.loads(cached_data)
            
            if input_data:
                # Process the data
                result = await pipeline['processor_func'](input_data)
                
                # Publish result if output topic specified
                if pipeline['output_topic'] and result:
                    await self.publish_event(
                        pipeline['output_topic'],
                        result,
                        key=pipeline['name']
                    )
                
                # Update metrics
                pipeline['processed_count'] += 1
                
        except Exception as e:
            logger.error(f"Error executing pipeline {pipeline['name']}: {e}")
            pipeline['error_count'] += 1
    
    async def get_live_game_stream(self, game_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Get a live stream of game events.
        
        Args:
            game_id: Game identifier
            
        Yields:
            Real-time game events
        """
        logger.info(f"Starting live game stream for {game_id}")
        
        # Subscribe to game-specific Redis channel
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe(f"game:{game_id}")
        
        try:
            while self.is_running:
                message = await pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    try:
                        event_data = json.loads(message['data'])
                        yield event_data
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in game stream: {message['data']}")
                
        except Exception as e:
            logger.error(f"Error in live game stream: {e}")
        finally:
            await pubsub.unsubscribe(f"game:{game_id}")
            await pubsub.close()
    
    async def publish_game_event(
        self,
        game_id: str,
        event_type: str,
        event_data: Dict[str, Any]
    ):
        """
        Publish a game-specific event.
        
        Args:
            game_id: Game identifier
            event_type: Type of event (score, foul, timeout, etc.)
            event_data: Event details
        """
        enriched_event = {
            'game_id': game_id,
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            **event_data
        }
        
        # Publish to Kafka topic
        await self.publish_event('game_events', enriched_event, key=game_id)
        
        # Publish to Redis channel for real-time subscribers
        await self.redis_client.publish(
            f"game:{game_id}",
            json.dumps(enriched_event)
        )
    
    async def get_processing_metrics(self) -> Dict[str, Any]:
        """Get stream processing metrics."""
        metrics = {
            'is_running': self.is_running,
            'pipelines': {},
            'redis_info': {},
            'kafka_info': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Pipeline metrics
        for name, pipeline in self.processing_pipelines.items():
            metrics['pipelines'][name] = {
                'processed_count': pipeline['processed_count'],
                'error_count': pipeline['error_count'],
                'error_rate': pipeline['error_count'] / max(1, pipeline['processed_count']),
                'created_at': pipeline['created_at']
            }
        
        # Redis metrics
        try:
            redis_info = await self.redis_client.info()
            metrics['redis_info'] = {
                'connected_clients': redis_info.get('connected_clients', 0),
                'used_memory': redis_info.get('used_memory_human', '0B'),
                'keyspace_hits': redis_info.get('keyspace_hits', 0),
                'keyspace_misses': redis_info.get('keyspace_misses', 0)
            }
        except Exception as e:
            logger.warning(f"Could not get Redis metrics: {e}")
        
        return metrics
    
    async def _update_processing_metrics(self, topic: str):
        """Update processing metrics for a topic."""
        metric_key = f"metrics:topic:{topic}"
        current_count = await self.redis_client.get(metric_key) or 0
        await self.redis_client.setex(
            metric_key,
            3600,  # 1 hour TTL
            int(current_count) + 1
        )
    
    async def _store_event_in_db(self, topic: str, event_data: Dict[str, Any]):
        """Store event in database for persistence."""
        try:
            # This would use the actual database connection
            # For now, just log the storage
            logger.debug(f"Storing event in DB: {topic} - {event_data.get('event_type')}")
        except Exception as e:
            logger.error(f"Failed to store event in database: {e}")
    
    async def _monitor_health(self):
        """Monitor stream processor health."""
        while self.is_running:
            try:
                # Check Redis connection
                await self.redis_client.ping()
                
                # Check Kafka producer
                if self.kafka_producer:
                    # Kafka producer health check would go here
                    pass
                
                # Update health status
                health_status = {
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'redis_connected': True,
                    'kafka_connected': self.kafka_producer is not None
                }
                
                await self.redis_client.setex(
                    'health:stream_processor',
                    60,  # 1 minute TTL
                    json.dumps(health_status)
                )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                
                # Update unhealthy status
                health_status = {
                    'status': 'unhealthy',
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
                
                try:
                    await self.redis_client.setex(
                        'health:stream_processor',
                        60,
                        json.dumps(health_status)
                    )
                except:
                    pass  # Redis might be down
                
                await asyncio.sleep(10)  # Check more frequently when unhealthy
    
    @cache_async_result(ttl=60)
    async def get_stream_summary(self) -> Dict[str, Any]:
        """Get summary of stream processing status."""
        return {
            'is_running': self.is_running,
            'total_pipelines': len(self.processing_pipelines),
            'total_handlers': len(self.event_handlers),
            'last_updated': datetime.now().isoformat()
        } 