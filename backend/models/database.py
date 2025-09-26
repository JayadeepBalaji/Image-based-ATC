import sqlite3
import aiosqlite
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import asyncio

class Database:
    """
    SQLite database handler for storing animal classification records.
    """
    
    def __init__(self, db_path: str = "../data/atc_database.db"):
        self.db_path = db_path
        self.connected = False
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    async def init_db(self):
        """Initialize the database with required tables."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Create classifications table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS classifications (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        animal_name TEXT,
                        species TEXT NOT NULL,
                        original_filename TEXT,
                        features TEXT,  -- JSON string of extracted features
                        classification_result TEXT,  -- JSON string of classification results
                        productivity_class TEXT,
                        confidence REAL,
                        overall_score REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create features table for detailed analysis
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS animal_features (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        classification_id TEXT,
                        feature_name TEXT,
                        feature_value REAL,
                        feature_score REAL,
                        FOREIGN KEY (classification_id) REFERENCES classifications (id)
                    )
                """)
                
                # Create system_stats table for tracking usage
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS system_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT,
                        total_classifications INTEGER DEFAULT 0,
                        high_productivity_count INTEGER DEFAULT 0,
                        medium_productivity_count INTEGER DEFAULT 0,
                        low_productivity_count INTEGER DEFAULT 0,
                        average_score REAL DEFAULT 0.0,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                await db.commit()
                self.connected = True
                print("Database initialized successfully!")
                
        except Exception as e:
            print(f"Error initializing database: {e}")
            self.connected = False
    
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self.connected
    
    async def save_classification(self, result: Dict) -> bool:
        """
        Save a classification result to the database.
        
        Args:
            result: Classification result dictionary
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Extract main data
                classification_id = result.get('id')
                timestamp = result.get('timestamp')
                animal_name = result.get('animal_name')
                species = result.get('species')
                original_filename = result.get('original_filename')
                features = json.dumps(result.get('features', {}))
                classification_result = json.dumps(result.get('classification', {}))
                
                classification_data = result.get('classification', {})
                productivity_class = classification_data.get('productivity_class')
                confidence = classification_data.get('confidence', 0.0)
                overall_score = classification_data.get('overall_score', 0.0)
                
                # Insert main classification record
                await db.execute("""
                    INSERT INTO classifications 
                    (id, timestamp, animal_name, species, original_filename, 
                     features, classification_result, productivity_class, confidence, overall_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    classification_id, timestamp, animal_name, species, original_filename,
                    features, classification_result, productivity_class, confidence, overall_score
                ))
                
                # Insert detailed features
                trait_scores = classification_data.get('trait_scores', {})
                for feature_name, feature_data in trait_scores.items():
                    if isinstance(feature_data, dict):
                        feature_value = feature_data.get('value', 0.0)
                        feature_score = feature_data.get('score', 0.0)
                        
                        await db.execute("""
                            INSERT INTO animal_features 
                            (classification_id, feature_name, feature_value, feature_score)
                            VALUES (?, ?, ?, ?)
                        """, (classification_id, feature_name, feature_value, feature_score))
                
                await db.commit()
                
                # Update system stats
                await self.update_system_stats()
                
                return True
                
        except Exception as e:
            print(f"Error saving classification: {e}")
            return False
    
    async def get_history(self, limit: int = 10) -> List[Dict]:
        """
        Get classification history.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of classification records
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                cursor = await db.execute("""
                    SELECT id, timestamp, animal_name, species, original_filename,
                           productivity_class, confidence, overall_score,
                           features, classification_result
                    FROM classifications 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                rows = await cursor.fetchall()
                
                history = []
                for row in rows:
                    record = {
                        'id': row['id'],
                        'timestamp': row['timestamp'],
                        'animal_name': row['animal_name'],
                        'species': row['species'],
                        'original_filename': row['original_filename'],
                        'classification': {
                            'productivity_class': row['productivity_class'],
                            'confidence': row['confidence'],
                            'overall_score': row['overall_score']
                        }
                    }
                    
                    # Parse JSON fields if needed
                    try:
                        if row['features']:
                            record['features'] = json.loads(row['features'])
                        if row['classification_result']:
                            full_classification = json.loads(row['classification_result'])
                            record['classification'].update(full_classification)
                    except:
                        pass  # Keep basic data if JSON parsing fails
                    
                    history.append(record)
                
                return history
                
        except Exception as e:
            print(f"Error getting history: {e}")
            return []
    
    async def get_statistics(self) -> Dict:
        """
        Get system statistics.
        
        Returns:
            Dictionary of system statistics
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Get basic counts
                cursor = await db.execute("""
                    SELECT 
                        COUNT(*) as total_classifications,
                        SUM(CASE WHEN productivity_class = 'High' THEN 1 ELSE 0 END) as high_count,
                        SUM(CASE WHEN productivity_class = 'Medium' THEN 1 ELSE 0 END) as medium_count,
                        SUM(CASE WHEN productivity_class = 'Low' THEN 1 ELSE 0 END) as low_count,
                        AVG(overall_score) as average_score,
                        AVG(confidence) as average_confidence
                    FROM classifications
                """)
                
                row = await cursor.fetchone()
                
                stats = {
                    'total_classifications': row['total_classifications'] or 0,
                    'high_productivity_count': row['high_count'] or 0,
                    'medium_productivity_count': row['medium_count'] or 0,
                    'low_productivity_count': row['low_count'] or 0,
                    'average_score': round(row['average_score'] or 0, 2),
                    'average_confidence': round(row['average_confidence'] or 0, 3),
                }
                
                # Calculate percentages
                total = stats['total_classifications']
                if total > 0:
                    stats['high_productivity_percentage'] = round((stats['high_productivity_count'] / total) * 100, 1)
                    stats['medium_productivity_percentage'] = round((stats['medium_productivity_count'] / total) * 100, 1)
                    stats['low_productivity_percentage'] = round((stats['low_productivity_count'] / total) * 100, 1)
                else:
                    stats['high_productivity_percentage'] = 0
                    stats['medium_productivity_percentage'] = 0
                    stats['low_productivity_percentage'] = 0
                
                # Get species breakdown
                cursor = await db.execute("""
                    SELECT species, COUNT(*) as count
                    FROM classifications
                    GROUP BY species
                """)
                
                species_data = await cursor.fetchall()
                species_stats = {}
                for row in species_data:
                    species_stats[row['species']] = row['count']
                
                stats['species_breakdown'] = species_stats
                
                # Get recent activity (last 7 days)
                cursor = await db.execute("""
                    SELECT DATE(timestamp) as date, COUNT(*) as daily_count
                    FROM classifications
                    WHERE DATE(timestamp) >= DATE('now', '-7 days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                """)
                
                recent_activity = await cursor.fetchall()
                stats['recent_activity'] = [{'date': row['date'], 'count': row['daily_count']} for row in recent_activity]
                
                return stats
                
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {
                'total_classifications': 0,
                'high_productivity_count': 0,
                'medium_productivity_count': 0,
                'low_productivity_count': 0,
                'average_score': 0,
                'average_confidence': 0
            }
    
    async def update_system_stats(self):
        """Update daily system statistics."""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            async with aiosqlite.connect(self.db_path) as db:
                # Get today's stats
                cursor = await db.execute("""
                    SELECT 
                        COUNT(*) as total_today,
                        SUM(CASE WHEN productivity_class = 'High' THEN 1 ELSE 0 END) as high_today,
                        SUM(CASE WHEN productivity_class = 'Medium' THEN 1 ELSE 0 END) as medium_today,
                        SUM(CASE WHEN productivity_class = 'Low' THEN 1 ELSE 0 END) as low_today,
                        AVG(overall_score) as avg_score_today
                    FROM classifications
                    WHERE DATE(timestamp) = ?
                """, (today,))
                
                row = await cursor.fetchone()
                
                # Insert or update today's stats
                await db.execute("""
                    INSERT OR REPLACE INTO system_stats 
                    (date, total_classifications, high_productivity_count, 
                     medium_productivity_count, low_productivity_count, average_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    today,
                    row[0] or 0,  # total_today
                    row[1] or 0,  # high_today
                    row[2] or 0,  # medium_today
                    row[3] or 0,  # low_today
                    round(row[4] or 0, 2)  # avg_score_today
                ))
                
                await db.commit()
                
        except Exception as e:
            print(f"Error updating system stats: {e}")
    
    async def delete_classification(self, classification_id: str) -> bool:
        """
        Delete a classification record.
        
        Args:
            classification_id: ID of the classification to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Delete from features table first (foreign key constraint)
                await db.execute("""
                    DELETE FROM animal_features 
                    WHERE classification_id = ?
                """, (classification_id,))
                
                # Delete from main classifications table
                cursor = await db.execute("""
                    DELETE FROM classifications 
                    WHERE id = ?
                """, (classification_id,))
                
                await db.commit()
                
                # Check if any rows were affected
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"Error deleting classification: {e}")
            return False
    
    async def search_classifications(self, 
                                  species: Optional[str] = None,
                                  productivity_class: Optional[str] = None,
                                  min_score: Optional[float] = None,
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  limit: int = 50) -> List[Dict]:
        """
        Search classifications with filters.
        
        Args:
            species: Filter by species
            productivity_class: Filter by productivity class
            min_score: Minimum overall score
            start_date: Start date for filtering
            end_date: End date for filtering
            limit: Maximum number of results
            
        Returns:
            List of matching classification records
        """
        try:
            query = """
                SELECT id, timestamp, animal_name, species, original_filename,
                       productivity_class, confidence, overall_score
                FROM classifications 
                WHERE 1=1
            """
            params = []
            
            if species:
                query += " AND species = ?"
                params.append(species)
            
            if productivity_class:
                query += " AND productivity_class = ?"
                params.append(productivity_class)
            
            if min_score is not None:
                query += " AND overall_score >= ?"
                params.append(min_score)
            
            if start_date:
                query += " AND DATE(timestamp) >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND DATE(timestamp) <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                
                results = []
                for row in rows:
                    results.append({
                        'id': row['id'],
                        'timestamp': row['timestamp'],
                        'animal_name': row['animal_name'],
                        'species': row['species'],
                        'original_filename': row['original_filename'],
                        'productivity_class': row['productivity_class'],
                        'confidence': row['confidence'],
                        'overall_score': row['overall_score']
                    })
                
                return results
                
        except Exception as e:
            print(f"Error searching classifications: {e}")
            return []
    
    async def get_feature_analysis(self, feature_name: str) -> Dict:
        """
        Get analysis for a specific feature across all animals.
        
        Args:
            feature_name: Name of the feature to analyze
            
        Returns:
            Feature analysis statistics
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                cursor = await db.execute("""
                    SELECT 
                        AVG(feature_value) as avg_value,
                        MIN(feature_value) as min_value,
                        MAX(feature_value) as max_value,
                        AVG(feature_score) as avg_score,
                        COUNT(*) as sample_count
                    FROM animal_features 
                    WHERE feature_name = ?
                """, (feature_name,))
                
                row = await cursor.fetchone()
                
                if row and row['sample_count'] > 0:
                    return {
                        'feature_name': feature_name,
                        'average_value': round(row['avg_value'], 3),
                        'min_value': round(row['min_value'], 3),
                        'max_value': round(row['max_value'], 3),
                        'average_score': round(row['avg_score'], 1),
                        'sample_count': row['sample_count']
                    }
                else:
                    return {
                        'feature_name': feature_name,
                        'error': 'No data available for this feature'
                    }
                    
        except Exception as e:
            print(f"Error analyzing feature: {e}")
            return {'feature_name': feature_name, 'error': str(e)}